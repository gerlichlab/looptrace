package at.ac.oeaw.imba.gerlich.looptrace

package object syntax:
  object all extends SyntaxForAll

  trait SyntaxForAll
      extends BooleanRelatedSyntax,
        SyntaxForBifunctor,
        SyntaxForFunction,
        SyntaxForImagingChannel,
        SyntaxForImagingTimepoint,
        SyntaxForJson,
        SyntaxForPath,
        SyntaxForTry
