
def optimizeThreshold(args):

  (sweeper,
   windows,
   timestamps,
   anomalyScores) = args

  # First, get the sweep-scores for each row in each data set
  allAnomalyRows = []

  curAnomalyRows = sweeper.calcSweepScore(
    timestamps,
    anomalyScores,
    windows
  )
  allAnomalyRows.extend(curAnomalyRows)

  # Get scores by threshold for the entire corpus
  scoresByThreshold = sweeper.calcScoreByThreshold(allAnomalyRows)
  scoresByThreshold = sorted(
    scoresByThreshold,key=lambda x: x.score, reverse=True)
  bestParams = scoresByThreshold[0]

  print(scoresByThreshold)
  print(("Optimizer found a max score of {} with anomaly threshold {}.".format(
    bestParams.score, bestParams.threshold
  )))
  return {
    "threshold": bestParams.threshold,
    "score": bestParams.score
  }
