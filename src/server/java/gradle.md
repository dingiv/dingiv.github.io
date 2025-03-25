# gradle
新一代java构建工具，在安卓开发中被默认使用。
1. 使用IDEA无须安装gradle，但是需要配置gradle的user_home
   ```groovy
   // windows环境变量
   GRADLE_USER_HOME
   ```
   之后的gradle的全局配置均在该目录下进行

2. 全局镜像库配置
   ```groovy
   // GRADLE_USER_HOME目录下，新建init.gradle文件
   allprojects{
       repositories {
           def ALIYUN_REPOSITORY_URL = 'https://maven.aliyun.com/repository/public'
           def ALIYUN_JCENTER_URL = 'https://maven.aliyun.com/repository/public'
           def ALIYUN_GOOGLE_URL = 'https://maven.aliyun.com/repository/google'
           def ALIYUN_GRADLE_PLUGIN_URL = 'https://maven.aliyun.com/repository/gradle-plugin'
           all { ArtifactRepository repo ->
               if(repo instanceof MavenArtifactRepository){
                   def url = repo.url.toString()
                   if (url.startsWith('https://repo1.maven.org/maven2/')) {
                       project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_REPOSITORY_URL."
                       remove repo
                   }
                   if (url.startsWith('https://jcenter.bintray.com/')) {
                       project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_JCENTER_URL."
                       remove repo
                   }
                   if (url.startsWith('https://dl.google.com/dl/android/maven2/')) {
                       project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_GOOGLE_URL."
                       remove repo
                   }
                   if (url.startsWith('https://plugins.gradle.org/m2/')) {
                       project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_GRADLE_PLUGIN_URL."
                       remove repo
                   }
               }
           }
           maven { url ALIYUN_REPOSITORY_URL }
           maven { url ALIYUN_JCENTER_URL }
           maven { url ALIYUN_GOOGLE_URL }
           maven { url ALIYUN_GRADLE_PLUGIN_URL }
       }
   }
   ```

3. 全局禁用奇怪的警告
   ```groovy
   // GRADLE_USER_HOME目录下，新建gradle.properties
   org.gradle.warning.mode=none
   ```

4. tomcat插件
   ```groovy
   // build.gradle
   plugins {
   	id "com.github.sahara3.tomcat-runner" version "0.2.1"
   }
   
   tomcat {
       version = 9.0
       port = 8080
       systemProperty 'your.custom.property', 'property-value'
   
       webapp(project(':myapp1')) {
           contextPath = '' // root context.
       }
   
       webapp project(':myapp2')
   
       webapp 'myapp3/build/libs/myapp3.war'
   
       webapp file('myapp4/build/libs/myapp4.war')
   }
   ```

5. groovy和java混编
   ```groovy
   // build.gradle
   plugins {
   	id "groovy"
   }
   
   sourceSets {
       main {
           java {
               srcDirs = [] // don't compile Java code twice
           }
           groovy {
               srcDirs = ['src/main/groovy', 'src/main/java']
           }
       }
   }
   // 或者，直接把java和groovy文件统一写在groovy文件夹下，不要java文件夹了
   ```
   
   

6. 常见的包
   ```groovy
   // build.gradle
   dependencies {
   	implementation localGroovy() //本地groovy
   	implementation 'org.apache.groovy:groovy:4.0.1'  //groovy
   	compileOnly 'javax.servlet:javax.servlet-api:4.0.1' //servlet
   	implementation 'mysql:mysql-connector-java:8.0.29' //JDBC
   	implementation 'org.mybatis:mybatis:3.5.9' //Mybatis
   	compileOnly 'javax.servlet.jsp:javax.servlet.jsp-api:2.3.3' //JSP
   	implementation 'commons-io:commons-io:2.11.0' //文件操作
   	compileOnly 'javax:javaee-api:8.0.1'
   }
   ``` 

7. Gradle重要概念
   + wrapper
   + closure
   + project
   + tasks
   + hooks
   + plugin。插件就是一段可重复使用的代码，插件分为对象插件和脚本插件，对象插件就是在build.gradle或其他类文件里的java类，它实现了Plugin\<Project\>接口，无须使用apply使用，如果是脚本或其他文件里的，需要写apply from:'${filePath}'，一般地脚本插件命名为'filename.gradle'，如果要使用脚本插件中定义的变量，需要在脚本插件中使用ext{}，进行闭包导出。