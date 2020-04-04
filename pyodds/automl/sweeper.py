
from collections import namedtuple
import logging
import math

logger = logging.getLogger(__name__)
AnomalyPoint = namedtuple(
  "AnomalyPoint",
  ["timestamp", "anomalyScore", "sweepScore", "windowName"]
)
ThresholdScore = namedtuple(
  "ThresholdScore",
  ["threshold", "score", "tp", "tn", "fp", "fn", "total"]
)


def sigmoid(x):
  """Standard sigmoid function."""
  return 1 / (1 + math.exp(-x))


def scaledSigmoid(relativePositionInWindow):

  if relativePositionInWindow > 3.0:
    # FP well behind window
    val = -1.0
  else:
    val = 2*sigmoid(-5*relativePositionInWindow) - 1.0

  return val


def prepAnomalyListForScoring(inputAnomalyList):

  return sorted(
    [x for x in inputAnomalyList if x.windowName != 'probationary'],
    key=lambda x: x.anomalyScore,
    reverse=True)

class Sweeper(object):

  def __init__(self, probationPercent=0.15, costMatrix=None):
    self.probationPercent = probationPercent

    self.tpWeight = 0
    self.fpWeight = 0
    self.fnWeight = 0

    if costMatrix is not None:
      self.setCostMatrix(costMatrix)


  def setCostMatrix(self, costMatrix):
    self.tpWeight = costMatrix["tpWeight"]
    self.fpWeight = costMatrix["fpWeight"]
    self.fnWeight = costMatrix["fnWeight"]


  def _getProbationaryLength(self, numRows):
    return min(
      math.floor(self.probationPercent * numRows),
      self.probationPercent * 5000
    )


  def _prepareScoreByThresholdParts(self, inputAnomalyList):
    scoreParts = {"fp": 0}
    for row in inputAnomalyList:
      if row.windowName not in ('probationary', None):
        scoreParts[row.windowName] = -self.fnWeight
    return scoreParts


  def calcSweepScore(
      self, timestamps, anomalyScores, windowLimits):

    assert len(timestamps) == len(anomalyScores), \
      "timestamps and anomalyScores should not be different lengths!"
    timestamps = list(timestamps)
    windowLimits = list(windowLimits)  # Copy because we mutate this list
    # The final list of anomaly points returned from this function.
    # Used for threshold optimization and scoring in other functions.
    anomalyList = []

    # One-time config variables
    maxTP = scaledSigmoid(-1.0)
    probationaryLength = self._getProbationaryLength(len(timestamps))

    # Iteration variables - these update as we iterate through the data
    curWindowLimits = None
    curWindowName = None
    curWindowWidth = None
    curWindowRightIndex = None
    prevWindowWidth = None
    prevWindowRightIndex = None

    for i, (curTime, curAnomaly) in enumerate(zip(timestamps, anomalyScores)):
      unweightedScore = None
      weightedScore = None

      # If not in a window, check if we've just entered one
      if windowLimits and curTime == windowLimits[0][0]:
        curWindowLimits = windowLimits.pop(0)

        curWindowRightIndex = timestamps.index(curWindowLimits[1])
        curWindowWidth = float(curWindowRightIndex -
                               timestamps.index(curWindowLimits[0]) + 1)


      # If in a window, score as if true positive
      if curWindowLimits is not None:
        positionInWindow = -(curWindowRightIndex - i + 1) / curWindowWidth
        unweightedScore = scaledSigmoid(positionInWindow)
        weightedScore = unweightedScore * self.tpWeight / maxTP

      # If outside a window, score as if false positive
      else:
        if prevWindowRightIndex is None:
          # No preceding window, so return score as is we were just really
          # far away from the nearest window.
          unweightedScore = -1.0
        else:
          numerator = abs(prevWindowRightIndex - i)
          denominator = float(prevWindowWidth - 1)
          positionPastWindow = numerator / denominator
          unweightedScore = scaledSigmoid(positionPastWindow)

        weightedScore = unweightedScore * self.fpWeight

      if i >= probationaryLength:
        pointWindowName = curWindowName
      else:
        pointWindowName = "probationary"

      point = AnomalyPoint(curTime, curAnomaly, weightedScore, pointWindowName)

      anomalyList.append(point)

      # If at right-edge of window, exit window.
      # This happens after processing the current point and appending it
      # to the list.
      if curWindowLimits is not None and curTime == curWindowLimits[1]:
        logger.debug("Exiting window: %s", curWindowName)
        prevWindowRightIndex = i
        prevWindowWidth = curWindowWidth
        curWindowLimits = None
        curWindowName = None
        curWindowWidth = None
        curWindowRightIndex = None

    return anomalyList


  def calcScoreByThreshold(self, anomalyList):
    """
    Find NAB scores for each threshold in `anomalyList`.

    @param anomalyList  (list) `AnomalyPoint` objects from `calcSweepScore()`

    @return (list)  List of `ThresholdScore` objects
    """
    scorableList = prepAnomalyListForScoring(anomalyList)
    scoreParts = self._prepareScoreByThresholdParts(scorableList)
    scoresByThreshold = []  # The final list we return

    # The current threshold above which an anomaly score is considered
    # an anomaly prediction. This starts above 1.0 so that all points
    # are skipped, which gives us a full false-negative score.
    curThreshold = 1.1

    # Initialize counts:
    # * every point in a window is a false negative
    # * every point outside a window is a true negative
    tn = sum(1 if x.windowName is None else 0 for x in scorableList)
    fn = sum(1 if x.windowName is not None else 0 for x in scorableList)
    tp = 0
    fp = 0
    print(" Points we need to beware of ",tn,fn)
    # Iterate through every data point, starting with highest anomaly scores
    # and working down. Whenever we reach a new anomaly score, we save the
    # current score and begin calculating the score for the new, lower
    # threshold. Every data point we iterate over is 'active' for the current
    # threshold level, so the point is either:
    #   * a true positive (has a `windowName`)
    #   * a false positive (`windowName is None`).
    for dataPoint in scorableList:
      # If we've reached a new anomaly threshold, store the current
      # threshold+score pair.
      if dataPoint.anomalyScore != curThreshold:
        curScore = sum(scoreParts.values())
        totalCount = tp + tn + fp + fn
        s = ThresholdScore(curThreshold, curScore, tp, tn, fp, fn, totalCount)
        scoresByThreshold.append(s)
        curThreshold = dataPoint.anomalyScore

      # Adjust counts
      if dataPoint.windowName is not None:
        tp += 1
        fn -= 1
      else:
        fp += 1
        tn -= 1

      if dataPoint.windowName is None:
        scoreParts["fp"] += dataPoint.sweepScore
      else:
        scoreParts[dataPoint.windowName] = max(
          scoreParts[dataPoint.windowName],
          dataPoint.sweepScore
        )

    # Make sure to save the score for the last threshold
    curScore = sum(scoreParts.values())
    totalCount = tp + tn + fp + fn
    s = ThresholdScore(curThreshold, curScore, tp, tn, fp, fn, totalCount)
    scoresByThreshold.append(s)

    return scoresByThreshold


  def scoreDataSet(
      self, timestamps, anomalyScores, windowLimits, threshold):

    anomalyList = self.calcSweepScore(
      timestamps, anomalyScores, windowLimits)
    scoresByThreshold = self.calcScoreByThreshold(anomalyList)

    matchingRow = None
    prevRow = None
    for thresholdScore in scoresByThreshold:
      if thresholdScore.threshold == threshold:
        matchingRow = thresholdScore
        break
      elif thresholdScore.threshold < threshold:
        matchingRow = prevRow
        break

      prevRow = thresholdScore

    # Return sweepScore for each row, to be added to score file
    print("anomalylist size",len(anomalyList), type(anomalyList[0]))
    return (
      [x.sweepScore for x in anomalyList],
      matchingRow
    )
