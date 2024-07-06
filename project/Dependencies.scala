import sbt._

/** Dependencies for the project */
object Dependencies {
    /** gerlichlab dependency declaration helper */
    object Gerlib {
        def getModuleId(name: String) = groupId %% artifact(name) % version
        private val groupId = "com.github.gerlichlab"
        private def artifact(module: String): String = s"gerlib-$module"
        private val version = "0.1.0"
    }

    /** Bundle data related to getting ModuleID for an iron subproject. */
    object Iron {
        // NB: iron use in this project is indirect, but it must be on the classpath.

        def moduleId = getModuleID(None)
        def getModuleID(name: String): ModuleID = getModuleID(Some(name))
        private def getModuleID(name: Option[String]): ModuleID = 
            groupId %% ("iron" ++ name.fold("")("-" ++ _)) % version
        private val rootName = "iron"
        private val groupId = "io.github.iltotore"
        private val version = "2.6.0"
    }

    /* Versions */
    lazy val scalatestVersion = "3.2.18"
    
    /* Core libraries */
    lazy val catsCore = "org.typelevel" %% "cats-core" % "2.12.0"
    lazy val gerlibs = Seq(
        "io",
        "imaging",
        "numeric",
        "pan",
    ).map(Gerlib.getModuleId)
    lazy val iron = Iron.moduleId
    lazy val ironCats = Iron.getModuleID("cats")
    lazy val ironScalacheck = Iron.getModuleID("scalacheck")
    lazy val logging = Seq(
        "ch.qos.logback" % "logback-classic" % "1.5.6", 
        "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
    )
    lazy val mouse = "org.typelevel" %% "mouse" % "1.2.3"
    lazy val os = "com.lihaoyi" %% "os-lib" % "0.10.2"
    lazy val scalaCsv = "com.github.tototoshi" %% "scala-csv" % "1.3.10"
    lazy val scopt = "com.github.scopt" %% "scopt" % "4.1.0"
    lazy val uJson = "com.lihaoyi" %% "ujson" % "3.3.1"
    lazy val uPickle = "com.lihaoyi" %% "upickle" % "3.3.1"

    /* Test dependencies */
    lazy val gerlibTesting = Gerlib.getModuleId("testing")
    lazy val scalacheck = "org.scalacheck" %% "scalacheck" % "1.18.0"
    lazy val scalactic = "org.scalactic" %% "scalactic" % scalatestVersion
    lazy val scalatest = "org.scalatest" %% "scalatest" % scalatestVersion
    lazy val scalatestScalacheck = "org.scalatestplus" %% "scalacheck-1-17" % "3.2.18.0"

}
