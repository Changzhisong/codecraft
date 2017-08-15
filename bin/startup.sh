#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
APP_HOME=$basepath/..

JAVA=$JAVA_HOME/bin/java

JVM_OPT="-Xms64M -Xmx64M"
JVM_OPT="$JVM_OPT -Djava.library.path=$APP_HOME/bin"
JVM_OPT="$JVM_OPT -classpath"
JVM_OPT="$JVM_OPT $APP_HOME/bin/cdn.jar"
inputCaseFile=$1
resultFilePath=$2
$JAVA $JVM_OPT $JAVAENV com.filetool.main.Main $inputCaseFile  $resultFilePath 2>&1
exit
