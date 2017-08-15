#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
APP_HOME=$basepath

#编译
echo building...
MAKE_FILE=$APP_HOME/makelist.txt
cd "$APP_HOME/code/cdn/src"
javac -source 1.7 -target 1.7 -d $APP_HOME/code/cdn/bin -encoding UTF-8 @$MAKE_FILE

#打包
echo make jar...
cd "$APP_HOME/code/cdn/bin"
JAR_NAME=$APP_HOME/bin/cdn.jar
jar -cvf $JAR_NAME *

cd $APP_HOME

echo build jar success!
exit
