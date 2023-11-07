package at.ac.oeaw.imba.gerlich.looptrace

/** Helpers for working with the excellent uJson project */
object UJsonHelpers:
    /** Lift floating-point to JSON number, through floating-point. */
    given liftDouble: (Double => ujson.Num) = ujson.Num.apply

    /** Lift integer to JSON number, through floating-point. */
    given liftInt: (Int => ujson.Num) = z => liftDouble(z.toDouble)
    
    /** Lift floating-point to JSON number. */
    given liftStr: (String => ujson.Str) = ujson.Str.apply
    
    /** Lift mapping with text keys to JSON object. */
    def liftMap[V](using conv: V => ujson.Value): (Map[String, V] => ujson.Obj) = m => ujson.Obj.from(m.view.mapValues(conv).toList)

    /** Read given JSON file into value of target type. */
    def readJsonFile[A](jsonFile: os.Path)(using upickle.default.Reader[A]): A = upickle.default.read[A](os.read(jsonFile))
