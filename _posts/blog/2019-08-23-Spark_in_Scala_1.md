---
title: Spark in Scala (part 1)
subtitle: NLP with Stanford CoreNLP and LightGBM in MMLSpark
author: Ding Ma
layout: post
categories: [blog]
---

<span style="font-weight:bold;font-size:32px">0. Introduction</span>

In this blog, I will introduce how to use Stanford CoreNLP wrapper in Spark. It wraps Stanford CoreNLP annotators as Spark DataFrame functions following the simple APIs introduced in Stanford CoreNLP 3.7.0. Detailed information can be found [<span style="color:blue">here</span>](https://github.com/databricks/spark-corenlp). 

In order to use this package, remember to include the following parameter when running Spark.
```scala
--packages databricks:spark-corenlp:0.4.0-spark2.4-scala2.11 --jars stanford-corenlp-3.9.2-models.jar
```
The packages can be imported as
```scala
// original Stanford CoreNLP
import edu.stanford.nlp.models._
import edu.stanford.nlp.simple._
// Stanford CoreNLP wrapper for Spark
import com.databricks.spark.corenlp.functions._
```
After tokenization and lemmatization, I will train 5 different classification models including the LightGBM Classifier in `MMLSpark`.

---
<span style="font-weight:bold;font-size:32px">1. Dataset</span>

In this blog, I will use the StumbleUpon Evergreen Classification Challenge Dataset from [<span style="color:blue">Kaggle</span>](https://www.kaggle.com/c/stumbleupon/data). 

StumbleUpon is a user-curated web content discovery engine that recommends relevant, high quality pages and media to its users, based on their interests. Pages can either be classified as "ephemeral" or "evergreen". A high quality prediction of "ephemeral" or "evergreen" would greatly improve a recommendation system. 

The mission is to build a classifier which will evaluate a large set of URLs and label them as either evergreen or ephemeral.

The structure of the dataset looks like the following:
```scala
root
 |-- url: string (nullable = true)
 |-- urlid: integer (nullable = true)
 |-- boilerplate: string (nullable = true)
 |-- alchemy_category: string (nullable = true)
...
 |-- numwords_in_url: integer (nullable = true)
 |-- parametrizedLinkRatio: double (nullable = true)
 |-- spelling_errors_ratio: double (nullable = true)
 |-- label: integer (nullable = true)
```

In order to show how to use Stanford CoreNLP in Spark, I will simply the task as follows:

Instead of using all the features from the dataset, I will only use the "body" part from the webpage to make the prediction.

So the first step is to extract the body from the`boilerplate` column in the dataset.

---
<span style="font-weight:bold;font-size:32px">2. Preprocessing</span>

First, let's load the dataset as a Spark DataFrame.
```scala
// a UDF to reformulate the json column
val cleanUDF = udf((x:String) => x.replaceAll("\"\"","\"").tail.init.toLowerCase)

val df1raw = spark.read
.option("header", "true")
.option("inferSchema", "true")  // let Spark infer the schema
.option("delimiter","\t")  // set delimiter to be \t
.csv("data/10/stumbleupon/train.tsv")
.select("boilerplate","label")  // keep those 2 columns only 
.withColumn("cleanJson", cleanUDF($"boilerplate"))  // clean the boilerplate column
// preview
df1raw.show(5)
```
The DataFrame `df1raw` looks like the following.
```javascript
+--------------------+-----+--------------------+
|         boilerplate|label|           cleanJson|
+--------------------+-----+--------------------+
|"{""title"":""IBM...|    0|{"title":"ibm see...|
|"{""title"":""The...|    1|{"title":"the ful...|
|"{""title"":""Fru...|    1|{"title":"fruits ...|
|"{""title"":""10 ...|    1|{"title":"10 fool...|
|"{""title"":""The...|    0|{"title":"the 50 ...|
+--------------------+-----+--------------------+
only showing top 5 rows
```
Now we have a nicely formatted Json column, so we are ready to parse it using `from_json`.
```scala
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
// specify the structure of the Json column
val schema = StructType(List(
    StructField("title", StringType),
    StructField("body", StringType),
    StructField("url", StringType)))
val df1 = df1raw
.withColumn("jsonData", from_json($"cleanJson", schema))  // parse the Json column using from_json
.select($"jsonData.body",$"label")  // keep only body and label
.filter{!isnull($"body")}  // remove empty rows
.filter{length($"body")>20}  // remove rows that are too short (mainly rows with only whitespaces)
// preview
df1.show(5)
```
After parsing the Json column and extracting `body` from it, the DataFrame `df1` looks like the following.
```javascript
+--------------------+-----+
|                body|label|
+--------------------+-----+
|a sign stands out...|    0|
|and that can be c...|    1|
|apples the most p...|    1|
|there was a perio...|    1|
|jersey sales is a...|    0|
+--------------------+-----+
only showing top 5 rows
```
---
<span style="font-weight:bold;font-size:32px">3. NLP: Tokenization and Lemmatization</span>

Now we are ready to perform tokenization and lemmatization on the text. All available functions of Stanford CoreNLP can be found [<span style="color: blue">here</span>](https://stanfordnlp.github.io/CoreNLP/annotators.html). In this blog, I will only use `tokenize` and `lemma`. Both of them can be used simply as Spark DataFrame functions.
```scala
import com.databricks.spark.corenlp.functions._
val df1Token = df1
.withColumn("words",tokenize($"body"))  // tokenization
.withColumn("lemmas",lemma($"body"))  // lemmatization
// preview
df1Token.show(5)
```
After tokenization and lemmatization, the DataFrame `df1Token` looks like the following.
```javascript
+--------------------+-----+--------------------+--------------------+
|                body|label|               words|              lemmas|
+--------------------+-----+--------------------+--------------------+
|a sign stands out...|    0|[a, sign, stands,...|[a, sign, stand, ...|
|and that can be c...|    1|[and, that, can, ...|[and, that, can, ...|
|apples the most p...|    1|[apples, the, mos...|[apple, the, most...|
|there was a perio...|    1|[there, was, a, p...|[there, be, a, pe...|
|jersey sales is a...|    0|[jersey, sales, i...|[jersey, sale, be...|
+--------------------+-----+--------------------+--------------------+
only showing top 5 rows
```
Notice that `stands=>stand`, `apples=>apple`,`was=>be`, `sales=>sale` after lemmatization.

---
<span style="font-weight:bold;font-size:32px">4. Classification</span>

In order to test the models, I will first make a train/test split on the data. Also parsing is a bit slow, so we should also cache the result in order to train several models.
```scala
val Array(df1Train,df1Test) = df1Token.randomSplit(weights=Array(0.8,0.2),seed=42)
df1Train.cache()
df1Test.cache()
```
Now I will train 4 different models.
<span style="font-weight:bold;font-size:28px">4.1 Tokenization + Logistic Regression</span>
```scala
import org.apache.spark.ml.feature.{StopWordsRemover, HashingTF, IDF}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// without Stanford CoreNLP Tokenizer, you can also use 
// org.apache.spark.ml.feature.RegexTokenizer

// remove stopwords, using words column (Tokenization)
val remover1 = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")
// TF
val hashingTF1 = new HashingTF().setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(5000)
// IDF
val idf1 = new IDF().setInputCol("rawFeatures").setOutputCol("features")
// Logistic Regression
val lr1 = new LogisticRegression().setMaxIter(50).setRegParam(1)
// Pipeline
val pipeline1 = new Pipeline().setStages(Array(remover1, hashingTF1, idf1, lr1))
// Train the model
val model1 = pipeline1.fit(df1Train)
// Evaluator
val mce = new MulticlassClassificationEvaluator()
.setLabelCol("label")
.setPredictionCol("prediction")
.setMetricName("accuracy")
println(f"accuracy(train)\t ${mce.evaluate(model1.transform(df1Train))}")
println(f"accuracy(test)\t ${mce.evaluate(model1.transform(df1Test))}")
```
```javascript
accuracy(train)	 0.8421052631578947
accuracy(test)	 0.7798369162342476
```

<span style="font-weight:bold;font-size:28px">4.2 Lemmatization + Logistic Regression</span>
```scala
// using lemmas column (Lemmatization)
val remover2 = new StopWordsRemover().setInputCol("lemmas").setOutputCol("filteredWords")
val hashingTF2 = new HashingTF().setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(5000)
val idf2 = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val lr2 = new LogisticRegression().setMaxIter(50).setRegParam(1)
val pipeline2 = new Pipeline().setStages(Array(remover2, hashingTF2, idf2, lr2))
val model2 = pipeline2.fit(df1Train)
println(f"accuracy(train)\t ${mce.evaluate(model2.transform(df1Train))}")
println(f"accuracy(test)\t ${mce.evaluate(model2.transform(df1Test))}")
```
```javascript
accuracy(train)	 0.8464131859898858
accuracy(test)	 0.7805782060785768
```

<span style="font-weight:bold;font-size:28px">4.3 Lemmatization + Random Forest Classifier</span>
```scala
import org.apache.spark.ml.classification.RandomForestClassifier
val remover3 = new StopWordsRemover().setInputCol("lemmas").setOutputCol("filteredWords")
val hashingTF3 = new HashingTF().setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(5000)
val idf3 = new IDF().setInputCol("rawFeatures").setOutputCol("features")
// can further tune hyperparameters for better result
val rf3 = new RandomForestClassifier().setNumTrees(10).setMaxDepth(6)
val pipeline3 = new Pipeline().setStages(Array(remover3, hashingTF3, idf3, rf3))
val model3 = pipeline3.fit(df1Train)
println(f"accuracy(train)\t ${mce.evaluate(model3.transform(df1Train))}")
println(f"accuracy(test)\t ${mce.evaluate(model3.transform(df1Test))}")
```
```javascript
accuracy(train)	 0.8029593556845851
accuracy(test)	 0.7731653076352853
```

<span style="font-weight:bold;font-size:28px">4.4 Lemmatization + Gradient Boosted Trees Classifier</span>
```scala
import org.apache.spark.ml.classification.GBTClassifier
val remover4 = new StopWordsRemover().setInputCol("lemmas").setOutputCol("filteredWords")
val hashingTF4 = new HashingTF().setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(5000)
val idf4 = new IDF().setInputCol("rawFeatures").setOutputCol("features")
// can further tune hyperparameters for better result
val gbt4 = new GBTClassifier().setLossType("logistic").setMaxDepth(2).setMaxIter(50)
val pipeline4 = new Pipeline().setStages(Array(remover4, hashingTF4, idf4, gbt4))
val model4 = pipeline4.fit(df1Train)
println(f"accuracy(train)\t ${mce.evaluate(model4.transform(df1Train))}")
println(f"accuracy(test)\t ${mce.evaluate(model4.transform(df1Test))}")
```
```javascript
accuracy(train)	 0.7969657239183368
accuracy(test)	 0.7835433654558932
```

<span style="font-weight:bold;font-size:28px">4.5 Lemmatization + LightGBM Classifier</span>

I have already discussed how to use `LightGBM` in Python. Now I will use show how to use LightGBM for Scala in Spark. The documentation can be found [<span style="color: blue">here</span>](https://github.com/Azure/mmlspark). To use the packages, just simply include the following parameters when running spark.
```scala
--packages com.microsoft.ml.spark:mmlspark_2.11:0.18.1
```
Now we are ready to build and train a LightGBMClassifier model using this package.
```scala
import com.microsoft.ml.spark.lightgbm.LightGBMClassifier
val remover5 = new StopWordsRemover().setInputCol("lemmas").setOutputCol("filteredWords")
val hashingTF5 = new HashingTF().setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(5000)
val idf5 = new IDF().setInputCol("rawFeatures").setOutputCol("features")
// can further tune hyperparameters for better result
val lgbm5 = new LightGBMClassifier().setNumLeaves(31).setLearningRate(0.1).setNumIterations(100).setObjective("binary").setBaggingFraction(0.8)
val pipeline5 = new Pipeline().setStages(Array(remover5, hashingTF5, idf5, lgbm5))
val model5 = pipeline5.fit(df1Train)
println(f"accuracy(train)\t ${mce.evaluate(model5.transform(df1Train))}")
println(f"accuracy(test)\t ${mce.evaluate(model5.transform(df1Test))}")
```
```javascript
accuracy(train)	 0.9392584311533445
accuracy(test)	 0.8039364118092355
```
LightGBMClassifier model obtained the best result in a much short time comparing to the other 4 models above.

---
<span style="font-weight:bold;font-size:32px">5. Summary</span>

1. Spark sql function `from_json` provides a function to parse a String column in Json format.
2. Stanford CoreNLP wrapper in Spark provides simple Spark DataFrame functions `tokenize` and `lemma` for tokenization and lemmatization.
3. The `StopWordsRemover`, `HashingTF`, `IDF` function in `SparkML` Library can be used to remove stop words and compute TF-IDF of a list of tokens.
4. `MMLSpark` Library adds many deep learning and data science tools to the Spark ecosystem, including seamless integration of Spark Machine Learning pipelines with Microsoft Cognitive Toolkit (CNTK), LightGBM and OpenCV. 









<span style="font-weight:bold;font-size:32px">0. Introduction</span>
<span style="font-weight:bold;font-size:32px">0. Introduction</span>
<span style="font-weight:bold;font-size:32px">0. Introduction</span>
<span style="font-weight:bold;font-size:32px">0. Introduction</span>
<span style="font-weight:bold;font-size:32px">0. Introduction</span>
<span style="font-weight:bold;font-size:32px">0. Introduction</span>
<span style="font-weight:bold;font-size:32px">0. Introduction</span>