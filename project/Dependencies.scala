import sbt._

/** Dependencies for the project */
object Dependencies {
    /** Get a typelevel cats dependency specification. */
    object Cats {
        def getModuleId(name: String) = "org.typelevel" %% s"cats-$name" % "2.13.0"
    }

    /** gerlichlab dependency declaration helper */
    object Gerlib {
        def getModuleId(name: String) = groupId %% artifact(name) % version
        private val groupId = "com.github.gerlichlab"
        private def artifact(module: String): String = s"gerlib-$module"
        private val version = "0.4.1"
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

    /** Build ModuleID for a com.lihaoyi JSON-related project. */
    object HaoyiJson {
        def getModuleId(name: String): ModuleID = "com.lihaoyi" %% name % latestVersion
        private def latestVersion = "4.1.0"
    }

    object PureConfig {
        def getModuleId(name: String): ModuleID = "com.github.pureconfig" %% s"pureconfig-$name" % "0.17.8"
    }

    /* versions */
    lazy val scalatestVersion = "3.2.19"
    
    /* core libraries */
    lazy val catsCore = Cats.getModuleId("core")
    lazy val gerlibs = Seq(
        "graph",
        "io",
        "imaging",
        "numeric",
        "pan",
    ).map(Gerlib.getModuleId)
    lazy val logging = Seq(
        "ch.qos.logback" % "logback-classic" % "1.5.17", 
        "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
    )
    lazy val mouse = "org.typelevel" %% "mouse" % "1.3.2"
    lazy val squants = "org.typelevel" %% "squants" % "1.8.3"

    /* iron */
    lazy val iron = Iron.moduleId
    lazy val ironCats = Iron.getModuleID("cats")
    lazy val ironScalacheck = Iron.getModuleID("scalacheck")
    
    /* IO-related dependencies */
    lazy val fs2Csv = "org.gnieh" %% "fs2-data-csv" % "1.11.2"
    lazy val fs2IO = "co.fs2" %% "fs2-io" % "3.11.0"
    lazy val os = "com.lihaoyi" %% "os-lib" % "0.11.4"
    lazy val pureconfigCore = PureConfig.getModuleId("core")
    lazy val pureconfigGeneric = PureConfig.getModuleId("generic-scala3")
    lazy val scalaCsv = "com.github.tototoshi" %% "scala-csv" % "1.4.1"
    lazy val scopt = "com.github.scopt" %% "scopt" % "4.1.0"
    lazy val uJson = HaoyiJson.getModuleId("ujson")
    lazy val uPickle = HaoyiJson.getModuleId("upickle")

    /* testing-related dependencies */
    lazy val catsLaws = Cats.getModuleId("laws")
    lazy val gerlibTesting = Gerlib.getModuleId("testing")
    lazy val scalacheck = "org.scalacheck" %% "scalacheck" % "1.18.1"
    lazy val scalacheckOps = "com.rallyhealth" %% "scalacheck-ops_1" % "2.12.0"
    lazy val scalactic = "org.scalactic" %% "scalactic" % scalatestVersion
    lazy val scalatest = "org.scalatest" %% "scalatest" % scalatestVersion
    lazy val scalatestScalacheck = "org.scalatestplus" %% "scalacheck-1-18" % "3.2.19.0"

}
