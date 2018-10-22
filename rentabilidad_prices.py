import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import lit,create_map
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DateType, LongType, StringType
from itertools import chain
import sys


sc = pyspark.SparkContext("local[*]")
sqlContext = pyspark.sql.SQLContext(sc)

# Recogemos parametros
INPUT_CSV = sys.argv[1]  # 'gs://financials-data-bucket/data/prueba/dataSBG/price.csv'
OUTPUT_CSV = sys.argv[2]  # 'gs://financials-data-bucket/data/prueba/rentabilidad_prices2.csv'

print('Contexto creado')

priceDF = sqlContext.read.format('csv') \
  .options(header='true', inferSchema='true') \
  .load(INPUT_CSV)

print('Archivo cargado')

priceDF = priceDF.select( \
F.col('code').alias('CODE'), \
F.col('Price').alias('PRICE'), \
F.col('Date').alias('DATE'))
priceDF = priceDF.withColumn('YEAR', F.substring('DATE', 1, 4).cast(IntegerType()))

print('Dataset creado')

#Rentabilidad diaria
windowSpec = Window.partitionBy("CODE").orderBy(F.col('DATE').asc()).rowsBetween(-1,0)
priceDF = priceDF.withColumn('AUX', F.sum("price").over(windowSpec))
priceDF = priceDF.withColumn("RETURNS", (F.col("PRICE") - (F.col("AUX")-F.col("PRICE"))) / F.col("PRICE")).drop("AUX")

print('Rentabilidad diaria')

#Rentabilidad acumulada
wCY = Window.partitionBy("CODE", "YEAR").orderBy("DATE")
#Nos quedamos con el precio de cada companya en cada anyo el dia 1
dico = priceDF.withColumn("RminD",  F.row_number().over(wCY)).filter("RminD == 1").drop("DATE", "RminD")
df_dict = [{r['CODE'] + str(r['YEAR']): r['PRICE']} for r in dico.orderBy("CODE", "YEAR").collect()] #Es una lista
df_dict = dict((key,d[key]) for d in df_dict for key in d) #Transformo a diccionario
mapping_expr = create_map([F.lit(x) for x in chain(*df_dict.items())]) #Transformamos a un mapa
r_acuDF = priceDF.withColumn("CUMULATIVE_RETURNS", (F.col("PRICE") - mapping_expr.getItem(F.concat(F.col("CODE"), F.col("YEAR").cast(StringType())))) / mapping_expr.getItem(F.concat(F.col("CODE"), F.col("YEAR").cast(StringType()))))

print('Rentabilidad acumulada')

r_acuDF \
.write.format("com.databricks.spark.csv") \
.option("header", "true") \
.save(OUTPUT_CSV)

print('Fichero guardado')

sc.stop()