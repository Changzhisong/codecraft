2017年华为软件精英挑战赛64强源码
=====
看到中文readme是一件非常亲切的事情！仔细看完此文档即可完成第一个用例：
***
* 1、整体指引：
  * 1)使用build.sh编译后打包;
  * 2)如果编译成功会在bin路径下生成可执行二进制文件"cdn.jar"；
  * 3)使用如下格式调用并调试程序，例如：<br>
  	Linux环境：root> sh startup.sh /xxx/topo.txt /xxx/result.txt(说明：topo.txt是输入文件，result.txt是输出文件)<br>
     	ps 以上步骤用于本地调试代码<br>
  * 4)将源代码与makelist.txt更新后打包压缩到同一个压缩包下，压缩包格式为tar.gz或zip，即可上传答案。<br>
  	上传答案的目录格式为：<br>
	cdn.tar.gz(文件格式可以为tar.gz或者zip格式，文件名称可以自定义，但不要包含中文或空格)<br>
	├─-bin/	<br>
	└── code/<br>
	└── makelist.txt<br>
	├── build.sh		不可修改，否则会影响服务器打包编译<br>
	├── cdn_tar.sh<br>
	└── readme.txt	<br>

* 2、SDK目录结构：
SDK-java.zip<br>
├─bin/				 								     		二进制文件路径<br>
│			└── cdn.jar								            　  　  可执行的jar文件，由一键式打包脚本生<br>
│		└── startup.sh								   　　		Linux环境一键式执行脚本(本地测试脚本)<br>
├─code/														源代码路径<br>
│	└── cdn/<br>
│		├── bin/										    编译后创建的class文件<br>
│		└── src/										    java源代码路径<br>
│				└─com/<br>
│					├─filetool/<br>
│					│	├─main/<br>
│					│	│		Main.java					main函数源文件，不可修改<br>
│					│	└─util/<br>
│					│			FileUtil.java				提供读写文件等功能的源文件，不可修改<br>
│					│			LogUtil.java				提供日志记录功能的源文件，不可修改<br>
│					└─cacheserverdeploy/<br>
│						└─deploy/<br>
│								Deploy.java					你要写代码的源文件，需要修改<br>
├── build.sh												Linux环境编译脚本，生成cdn.jar，不可修改<br>
├── makelist.txt　										  　  需要编译的java文件，供脚本调用<br>
├── cdn_tar.sh								                打包脚本，生成cdn.tar.gz(可以不使用，直接在windows上修改源代码和makelist.txt压缩成zip格式文件)<br>
└── readme.txt											    你正在看的文件 -_-" 这不用介绍了吧<br>
	注意：<br>
	* 1)如果增加了源文件需要修改makelist.txt文件；<br>
	* 2)如果运行时提示startup.sh文件第13行错误，请按照你本地系统情况修改此行，此脚本只是用来方便调用，修改后不会影响在判题平台的判题。<br>

* 3、SDK代码说明：<br>
  我们已经提供了保姆式的服务，你只需要做：
	* 1)完成Deploy.java文件中的XXX方法；
	* 2)如果计算结果无解，则直接输出NA。<br>
  SDK已经实现了读取文件、按要求格式写文件以及打印开始和结束时间的功能。为了便于调试，SDK将读入的信息全部在屏幕输出，可根据自身的需要酌情删除此打印信息。<br>
  注意：读取文件功能是指，将图的信息文件和路径信息文件按行读取到内存，其在内存中的存储格式仍是字符串格式。因为这些信息以什么格式存储涉及到算法设计，这样做是为了不禁锢你的思路。<br>
  
* 4、重要提示：<br>
  初赛直接提交你修改和增加的代码源文件，在判题服务器端执行编译。因此需要特别注意：
	* 1、必须基于本SDK开发，否则会编译不过；
	* 2、在SDK的源文件中，只有Deploy.java和makelist.txt文件允许修改，不要修改其他文件，否则可能编译不通过；
	* 3、请使用jdk1.7版本进行开发。
