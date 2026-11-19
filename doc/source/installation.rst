Installation
------------

To install the package, please use the `pip`_ installation as
follows:

.. code:: sh

   pip install pyodds
   pip install git+git@github.com:datamllab/PyODDS.git

**Note:** PyODDS is only compatible with **Python 3.6** and above.

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   - pandas>=0.25.0
   - tensorflow==2.0.0b1
   - numpy>=1.16.4
   - seaborn>=0.9.0
   - torch>=1.1.0
   - luminol==0.4
   - tqdm>=4.35.0
   - matplotlib>=3.1.1
   - scikit_learn>=0.21.3

TDengine and Apache IoTDB clients are optional backend dependencies. Install
the selected backend with ``pip install 'pyodds[tdengine]'`` or
``pip install 'pyodds[iotdb]'``. The IoTDB integration targets the 2.x Python
client and uses the Tree data model.

To compile and package the JDBC driver source code, you should have a
Java jdk-8 or higher and Apache Maven 2.7 or higher installed. To
install openjdk-8 on Ubuntu:

.. code:: sh

   sudo apt-get install openjdk-8-jdk

To install Apache Maven on Ubuntu:

.. code:: sh

   sudo apt-get install maven

To install the TDengine as the back-end database service, please refer
to `this instruction`_.

To enable the Python client APIs for TDengine, please follow `this
handbook`_.

To connect the demo to IoTDB, run ``python demo.py --backend iotdb --host
<host> --port 6667 --user <user> --database <database> --table <device>`` and
enter the password at the prompt. The integration creates measurements ``a``
and ``b`` under ``root.<database>.<device>`` and closes the IoTDB session when
the demo finishes.

To insure the locale in config file is valid:

.. code:: sh

   sudo locale-gen "en_US.UTF-8"
   export LC_ALL="en_US.UTF-8"
   locale

To start the service after installation, in a terminal, use:

.. code:: sh

   taosd

.. _pip: https://pip.pypa.io/en/stable/installing/
.. _this instruction: https://www.taosdata.com/en/getting-started/#Install-from-Package
.. _this handbook: https://www.taosdata.com/en/documentation/connector/#Python-Connector
