---
title: Spark in Scala (part 2)
subtitle: NLP with MMLSpark
author: Ding Ma
layout: post
categories: [blog]
---

<span style="font-weight:bold;font-size:32px">0. Introduction</span>

In this blog, I will introduce how to use `MMLSpark` to perform some NLP tasks. The documentation of `MMLSpark` can be found [<span style="color: blue">here</span>](https://github.com/Azure/mmlspark). To use the packages, just simply include the following parameters when running Spark.
```javascript
--packages com.microsoft.ml.spark:mmlspark_2.11:0.18.1
```
---
<span style="font-weight:bold;font-size:32px">1. Dataset and Task</span>

I will use Crowdflower Search Results Relevance Dataset from [<span style="color: blue">Kaggle</span>](https://www.kaggle.com/c/crowdflower-search-relevance/data). The dataset includes the following features:
* id: Product id
* query: Search term used
* product_title: Product title
* product_description: The full product description
* median_relevance: Median relevance score by 3 raters. This value is an integer between 1 and 4. 
* relevance_variance: Variance of the relevance scores given by raters. 

The task is to predict the `median_relevance` using other features. To simplify the problem, I will only use `query` and `product_title` in this blog. Both of those two features are NLP features. I will treat this problem as a multi-class classification problem here.

---
<span style="font-weight:bold;font-size:32px">2. Load as a DataFrame</span>

The original data is not well formatted (a lot of commas in the text column). So instead of using `spark.read.csv`, I will read the file line by line using `scala.io.Source`.

```javascript
// using spark.read.csv leads to a DataFrame with a lot of bad entries
+---+--------------------+--------------------+--------------------+----------------+------------------+
| id|               query|       product_title| product_description|median_relevance|relevance_variance|
+---+--------------------+--------------------+--------------------+----------------+------------------+
|  1|bridal shower dec...|Accent Pillow wit...|"Red satin accent...|               1|                 0|
|  2|led christmas lights|Set of 10 Battery...|"Set of 10 Batter...|               4|                 0|
|  4|           projector|ViewSonic Pro8200...|                null|               4|             0.471|
|  5|           wine rack|Concept Houseware...|Like a silent and...|            null|              null|
+---+--------------------+--------------------+--------------------+----------------+------------------+
only showing top 4 rows
```
I will use regular expression to parse the file line by line as follows.
```scala
import org.apache.spark.sql.Row
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.types._
// use regex to extract legal lines
val pattern  = "^(\\d+),\"(.*)\",\"(.*)\",\"(.*)\",([1234]+),([\\.\\d]+)$".r
// file path
val filename = "data/rank_crowd_flower/train.csv"
// ArrayBuffer to store parsed lines
val lines = new ArrayBuffer[Row]()
for (line <- Source.fromFile(filename).getLines) {
    val matchGroup = pattern.findAllMatchIn(line)
    // for any valid line, add a Row into the Array
    // meanwhile, clean "query","title","description" column
    if (!matchGroup.isEmpty) {
        val subgroups = matchGroup.next.subgroups
        lines.append(Row(subgroups(0).toInt,subgroups(1).replaceAll("[^\\w\\d]"," ").replaceAll("\\s+"," "),
                         subgroups(2).replaceAll("[^\\w\\d]"," ").replaceAll("\\s+"," "),
                         subgroups(3).replaceAll("[^\\w\\d]"," ").replaceAll("\\s+"," "),
                         subgroups(4).toInt,subgroups(5).toFloat))
    }
}
// schema of DataFrame
val schema = new StructType(Array(
    StructField("id", IntegerType),
    StructField("query", StringType),
    StructField("title", StringType),
    StructField("description", StringType),
    StructField("med_relevance", IntegerType),
    StructField("var_relevance", FloatType)))
// transform from ArrayBuffer to DataFrame
val dfRaw = spark.createDataFrame(sc.parallelize(lines,2),schema=schema)
// preview
dfRaw.show(5)
```
Now the DataFrame looks much better.
```javascript
+---+--------------------+--------------------+--------------------+-------------+-------------+
| id|               query|               title|         description|med_relevance|var_relevance|
+---+--------------------+--------------------+--------------------+-------------+-------------+
|  1|bridal shower dec...|Accent Pillow wit...|Red satin accent ...|            1|          0.0|
|  2|led christmas lights|Set of 10 Battery...|Set of 10 Battery...|            4|          0.0|
|  4|           projector|ViewSonic Pro8200...|                    |            4|        0.471|
| 13|screen protector ...|ZAGG InvisibleShi...|Protect your most...|            4|          0.0|
| 17|   pots and pans set|Cook N Home Stain...|This ultimate pas...|            2|        0.632|
+---+--------------------+--------------------+--------------------+-------------+-------------+
only showing top 5 rows
```

<span style="font-weight:bold;font-size:32px">3. Preprocessing</span>

<span style="font-weight:bold;font-size:28px">3.1 Number of Common Words</span>

When product title or description contains the same words as the query, it is more likely to be relevant. So I will first create two features counting common words.

```scala
import org.apache.spark.sql.functions._
// UDF computing common words
def commonwordUDF = udf((x: String, y: String) => {
    val a = x.split(" ").toSet
    val b = y.split(" ").toSet
    a.intersect(b).size
})
// create two columns for common words in (query,title) and (query,description)
val dfRaw2 = dfRaw
.withColumn("qt",commonwordUDF($"query",$"title"))
.withColumn("qd",commonwordUDF($"query",$"description"))
```
<span style="font-weight:bold;font-size:28px">3.2 Train/Test Split</span>

In order to evaluate the model, I will split the DataFrame into a train set and a test set.
```scala
val Array(dfTrain,dfTest) = dfRaw2.randomSplit(weights=Array(0.8,0.2),seed=42)
```

<span style="font-weight:bold;font-size:28px">3.3 NLP using MMLSpark</span>

Using `TextFeaturizer` from `MMLSpark` library, we can perform both Stopwords Removing, TF-IDF and Bigram in one step.
```scala
import com.microsoft.ml.spark.featurize.text.TextFeaturizer
// choose top 3000 unigrams and top 1000 bigrams from both query and title
// Unigram + TF-IDF + StopWordsRemoving
val qFeaturizer1 = new TextFeaturizer()
.setInputCol("query").setOutputCol("qFeatures1")
.setToLowercase(true).setUseStopWordsRemover(true)
.setBinary(true).setUseIDF(true).setMinDocFreq(2)
.setNumFeatures(3000).fit(dfTrain)
val tFeaturizer1 = new TextFeaturizer()
.setInputCol("title").setOutputCol("tFeatures1")
.setToLowercase(true).setUseStopWordsRemover(true)
.setBinary(true).setUseIDF(true).setMinDocFreq(2)
.setNumFeatures(3000).fit(dfTrain)
// Bigrams + TF-IDF + StopWordsRemoving
val qFeaturizer2 = new TextFeaturizer()
.setInputCol("query").setOutputCol("qFeatures2")
.setToLowercase(true).setUseStopWordsRemover(true)
.setBinary(true).setUseIDF(true).setMinDocFreq(1)
.setUseNGram(true).setNGramLength(2)
.setNumFeatures(1000).fit(dfTrain)
val tFeaturizer2 = new TextFeaturizer()
.setInputCol("title").setOutputCol("tFeatures2")
.setToLowercase(true).setUseStopWordsRemover(true)
.setBinary(true).setUseIDF(true).setMinDocFreq(1)
.setUseNGram(true).setNGramLength(2)
.setNumFeatures(1000).fit(dfTrain)
// transform DataFrames
val dfTrain1 = tFeaturizer2.transform(tFeaturizer1.transform(qFeaturizer2.transform(qFeaturizer1.transform(dfTrain))))
.select($"qFeatures1",$"tFeatures1",$"qFeatures2",$"tFeatures2",$"med_relevance",$"qt",$"qd")
val dfTest1 = tFeaturizer2.transform(tFeaturizer1.transform(qFeaturizer2.transform(qFeaturizer1.transform(dfTest))))
.select($"qFeatures1",$"tFeatures1",$"qFeatures2",$"tFeatures2",$"med_relevance",$"qt",$"qd")
// preview
dfTrain1.show(5)
```
Now we will have 4 sparse vector columns, 2 integer columns and 1 label column.
```javascript
+--------------------+--------------------+--------------------+--------------------+-------------+---+---+
|          qFeatures1|          tFeatures1|          qFeatures2|          tFeatures2|med_relevance| qt| qd|
+--------------------+--------------------+--------------------+--------------------+-------------+---+---+
|(3000,[700,2134,2...|(3000,[816,995,11...|(1000,[344,427],[...|(1000,[64,65,188,...|            1|  0|  0|
|(3000,[1015,1813,...|(3000,[2,365,658,...|(1000,[165,951],[...|(1000,[20,161,211...|            2|  1|  2|
|(3000,[628,1506,2...|(3000,[20,860,987...|(1000,[759,800],[...|(1000,[250,455,68...|            3|  0|  0|
|(3000,[669,2090],...|(3000,[83,592,669...|(1000,[350],[5.77...|(1000,[6,257,346,...|            4|  0|  0|
|(3000,[404,2411],...|(3000,[404,856,16...|(1000,[982],[5.05...|(1000,[124,607,64...|            4|  0|  0|
+--------------------+--------------------+--------------------+--------------------+-------------+---+---+
only showing top 5 rows
```

<span style="font-weight:bold;font-size:28px">3.4 Assembling into a Vector</span>

In order to perform machine learning task, we will need to assemble all feature columns into a single vector column as follows.
```scala
import org.apache.spark.ml.feature.VectorAssembler
// VectorAssembler
val assembler = new VectorAssembler()
.setInputCols(Array("qFeatures1", "tFeatures1","qFeatures2", "tFeatures2","qt","qd"))
.setOutputCol("features")
// transform DataFrames, changing label from 1-4 into 0-3 for later use
val dfTrain2 = assembler.transform(dfTrain1).withColumn("label",$"med_relevance"-1)
val dfTest2 = assembler.transform(dfTest1).withColumn("label",$"med_relevance"-1)
// cache DataFrames
dfTrain2.cache()
dfTest2.cache()
```
---
<span style="font-weight:bold;font-size:32px">4. Training Models</span>

In order to show how to use ensemble modeling, I will train 3 different models and ensemble them as a single ensemble model.

```scala
import org.apache.spark.ml.classification.LogisticRegression
import com.microsoft.ml.spark.lightgbm.LightGBMClassifier
import org.apache.spark.ml.classification.NaiveBayes

// model1: Logistic Regression
val lr = new LogisticRegression().setMaxIter(50).setRegParam(0.1)
val lrModel = lr.fit(dfTrain2)

// model2: LightGBM Classifier
val lightGBM = new LightGBMClassifier()
.setNumLeaves(31).setLearningRate(0.5).setNumIterations(150)
.setObjective("multiclass")
.setBaggingFraction(0.8).setBaggingFreq(2).setFeatureFraction(0.3)
val lightGBMModel = lightGBM.fit(dfTrain2)

// model3: Naive Bayes Classifier
val nb = new NaiveBayes().setSmoothing(5.0)
val nbModel = nb.fit(dfTrain2)
```
---
<span style="font-weight:bold;font-size:32px">5. Evaluation</span>

<span style="font-weight:bold;font-size:28px">5.1 Two Metrics</span>

Before ensembling them, I will first evaluate model performances using two metrics. 
1. accuracy
2. [<span style="color: blue">quadratic weighted kappa</span>](https://en.wikipedia.org/wiki/Cohen's_kappa): the original Kaggle competition uses this metric. Cohen's kappa coefficient is a statistic that is used to measure inter-rater reliability (and also Intra-rater reliability) for qualitative (categorical) items. It is generally thought to be a more robust measure than simple percent agreement calculation, as it takes into account the possibility of the agreement occurring by chance. The weighted kappa allows disagreements to be weighted differently and is especially useful when codes are ordered.

Since the quadratic weighted kappa is not implemented in Spark, I wrote it myself.
```scala
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Matrix

def computeKappa(c: Matrix, pal: RDD[(Double, Double)]): Double = {
    // confusion matrix
    val o = c.rowIter.toArray.map{_.toArray}
    val n = o.size
    // weight
    val w = Array.fill(n,n)(0.0)
    // E
    val e = Array.fill(n,n)(0.0)    
    for (i <- 0 until n; j <- 0 until n) {w(i)(j) = math.pow(i-j,2)/math.pow(n-1,2)}
    val aHist = Array.fill(n)(0.0)
    for (y <- pal.map{x=>(x._1,1)}.reduceByKey(_+_).sortBy(_._1).collect()) {aHist(y._1.toInt) = y._2}
    val bHist = Array.fill(n)(0.0)
    for (y <- pal.map{x=>(x._2,1)}.reduceByKey(_+_).sortBy(_._1).collect()) {bHist(y._1.toInt) = y._2}
    for (i <- 0 until n; j <- 0 until n) {e(i)(j) = aHist(i)*bHist(j)}
    val osum = o.map{_.sum}.sum
    val esum = e.map{_.sum}.sum
    var num=0.0
    var den=0.0
    for (i <- 0 until n; j <- 0 until n) {
        num += w(i)(j)*o(i)(j)/osum
        den += w(i)(j)*e(i)(j)/esum}
    1.0 - (num/den)
}
```
<span style="font-weight:bold;font-size:28px">5.2 Evaluation Results</span>
```scala
import org.apache.spark.mllib.evaluation.MulticlassMetrics
// by setting model = lrModel, lightGBMModel, nbModel
// and setting df = dfTrain2, dfTest2
val pal = lrModel.transform(df).select($"label".cast("Double"),$"prediction").rdd.map{x => (x.getDouble(0),x.getDouble(1))}
val metrics = new MulticlassMetrics(pal)
println(metrics.accuracy)
println(computeKappa(metrics.confusionMatrix,pal))
```
```javascript
// accuracy(train)
0.9163672002613525  // lrModel
0.8691604050963737  // lightGBMModel
0.7641293694870958  // nbModel

// accuracy(test)
0.6565589980224127  // lrModel
0.6235992089650626  // lightGBMModel
0.6394199077125906  // nbModel


// quadratic weighted kappa(train)
0.8941728860121952  // lrModel
0.8297456844979645  // lightGBMModel
0.6830308335488613  // nbModel

// quadratic weighted kappa(test)
0.49524477476700157  // lrModel
0.42080421530244017  // lightGBMModel
0.48092931399223104  // nbModel
```
We can see that we have already obtained a descent result (but overfitting very hard) using only `query` and `title` columns.

---
<span style="font-weight:bold;font-size:32px">6. Ensembling Model</span>

The above 3 models can be ensembled in the following 2 different ways:
* take a majority vote among the predictions
* take an average over the predicted probabilities

I will only present the evaluations on test set here.
```javascript
// accuracy(test)
0.6704021094264997  // majority vote
0.6591957811470006  // average probability


// quadratic weighted kappa(test)
0.5247437131596568  // majority vote
0.5119220483945226  // average probability
```
We can see that both methods improve the result quite a bit.

<span style="font-weight:bold;font-size:32px">7. Further Improvement</span>

In order to further improve the result, we can
1. instead of using `query` and `title` columns only, `description` can also be used
2. text can be parsed more carefully (such as `lemmatization`)
3. word vectors (such as `word2vec`, `doc2vec`, `GloVe`) can be used instead of `TF-IDF`
4. all the models above overfit quite hard, so a dimension reduction technique (such as `PCA`) can be used to reduce the dimension of the feature space
5. model hyperparameters can be fine-tuned to achieve better result 
6. ensemble a large variety kinds of models instead of using only 3 models as above