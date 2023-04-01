package com.ibda.spark.regression;

import cn.hutool.core.util.ReflectUtil;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 回归模型数据列设置
 * StringIndex转换，转换后形成新的Category变量，可能作为features中的一列，也可能成为label列
 */
public class ModelColumns implements Serializable {

    private static final String REGRESSION_FEATURES_VECTOR = "features_vector";
    private static final String REGRESSION_FEATURES_DEFAULT = "features";
    private static final String VECTOR_SUFFIX = "_vector";
    private static final String INDEX_SUFFIX = "_index";
    private static final String IMPUTED_SUFFIX = "_imputed";
    public static final String PCA_SUFFIX = "_pca";
    public static final String SCALED_SUFFIX = "_scaled";
    public static final String TOTAL_CATEGORY_COUNT = "total_category_count";

    public static ModelColumns MODEL_COLUMNS_DEFAULT =  new ModelColumns();

    String[] noneCategoryFeatures ; //特征属性列表
    String[] categoryFeatures;   //特征属性中的分类属性，需要进行OneHotEncoder处理
    String[] stringCategoryFeatures ;  //使用字符串类型的分类属性，需要转换为其索引号，按照"frequencyDesc"方式编码
    String featuresCol = REGRESSION_FEATURES_DEFAULT;
    String labelCol = "label";         //观察结果标签列
    String predictCol = "prediction";       //预测结果标签列
    String probabilityCol = "probability";   //或然列，逻辑回归时的概率值
    String weightCol = null; //权重列
    String[] additionCols = null; //附加列，参与建模的非特征列，建模前转换需要保留该列
    /**
     * 使用缺省值
     */
    private ModelColumns(){
        super();
    }

    /**
     *  @param noneCategoryFeatures
     * @param categoryFeatures
     * @param labelCol
     */
    public ModelColumns(String[] noneCategoryFeatures, String[] categoryFeatures, String labelCol){
        this(noneCategoryFeatures,categoryFeatures,null,labelCol);
    }

    /**
     *  @param noneCategoryFeatures    非编码字段
     * @param categoryFeatures         需要进行独热编码处理的分类字段，如果无需处理，则直接进入noneCategoryFeatures
     * @param stringCategoryFeatures   需要进行索引化处理的字符串类型的分类变量，索引后如需继续做独热编码，该字段需要在
     *                                 categoryFeatures中，否则需在noneCategoryFeatures中
     *
     * @param labelCol                 标签列，预测分类或预测结果列
     */
    public ModelColumns(String[] noneCategoryFeatures, String[] categoryFeatures, String[] stringCategoryFeatures, String labelCol){
        this.noneCategoryFeatures = noneCategoryFeatures;
        this.categoryFeatures = categoryFeatures;
        this.stringCategoryFeatures = stringCategoryFeatures;
        this.labelCol = labelCol;
        this.featuresCol = REGRESSION_FEATURES_VECTOR;
    }

    /**
     *
     * @param noneCategoryFeatures
     * @param categoryFeatures
     * @param labelCol
     * @param predictCol
     * @param probabilityCol
     */
    public ModelColumns(String[] noneCategoryFeatures, String[] categoryFeatures, String labelCol,
                        String predictCol, String probabilityCol) {
        this(noneCategoryFeatures, categoryFeatures, null, labelCol, predictCol, probabilityCol);
    }

    /**
     *  @param noneCategoryFeatures
     * @param categoryFeatures
     * @param stringCategoryFeatures
     * @param labelCol
     * @param predictCol
     * @param probabilityCol
     */
    public ModelColumns(String[] noneCategoryFeatures, String[] categoryFeatures, String[] stringCategoryFeatures, String labelCol,
                        String predictCol, String probabilityCol) {
        this(noneCategoryFeatures,categoryFeatures,stringCategoryFeatures,labelCol);
        this.predictCol = predictCol;
        this.probabilityCol = probabilityCol;
        this.featuresCol = REGRESSION_FEATURES_VECTOR;
    }

    public String getFeaturesCol(){
        return featuresCol;
    }

    public void setFeaturesCol(String featuresCol) {
        this.featuresCol = featuresCol;
    }

    public String[] getNoneCategoryFeatures() {
        return noneCategoryFeatures;
    }

    public String[] getCategoryFeatures() {
        return categoryFeatures;
    }

    public String getLabelCol() {
        return labelCol;
    }

    public String getPredictCol() {
        return predictCol;
    }

    public String getProbabilityCol() {
        return probabilityCol;
    }

    public String getWeightCol() {
        return weightCol;
    }

    public void setWeightCol(String weightCol) {
        this.weightCol = weightCol;
    }

    public String[] getAdditionCols() {
        return additionCols;
    }

    public void setAdditionCols(String[] additionCols) {
        this.additionCols = additionCols;
    }

    @Override
    public String toString() {
        return "ModelColumns{" +
                "noneCategoryFeatures=" + Arrays.toString(noneCategoryFeatures) +
                ", categoryFeatures=" + Arrays.toString(categoryFeatures) +
                ", stringCategoryFeatures=" + Arrays.toString(stringCategoryFeatures) +
                ", featuresCol='" + featuresCol + '\'' +
                ", labelCol='" + labelCol + '\'' +
                ", predictCol='" + predictCol + '\'' +
                ", probabilityCol='" + probabilityCol + '\'' +
                ", weightCol='" + weightCol + '\'' +
                ", additionCols=" + Arrays.toString(additionCols) +
                '}';
    }


    /**
     * 根据数据集训练PipelineModel，Pipeline和数据集相关，最多包括StringIndexer、OneHotEncoder、VectorAssembler三个步骤，
     * 其中StringIndexer使用alphabetAsc排序自动生成Index，如果需要外部的编码表，需要在外面自行处理
     * @param trainingData
     * @return
     */
    public PipelineModel fit(Dataset<Row> trainingData) {
        return fit(trainingData, false);
    }

    /**
     * 根据数据集训练PipelineModel，Pipeline和数据集相关，最多包括StringIndexer、OneHotEncoder、VectorAssembler三个步骤，
     * 其中StringIndexer使用alphabetAsc排序自动生成Index，如果需要外部的编码表，需要在外面自行处理
     * @param trainingData
     * @param scaleByMinMax
     * @return
     */
    public PipelineModel fit(Dataset<Row> trainingData, boolean scaleByMinMax) {
        //计算最后的总特征数,非分类特征直接计算，分类特征按照OneHotEncoder,dropLast计算
        long total_features_count = 0;
        if (categoryFeatures != null) {
            total_features_count = countDistinctCategoryFeatures(trainingData).get(TOTAL_CATEGORY_COUNT)-categoryFeatures.length;
        }
        if (noneCategoryFeatures != null) {
            total_features_count = total_features_count + noneCategoryFeatures.length;
        }
        //StringIndexer的fit样本自动编码 alphabetAsc
        List<PipelineStage> stageList = new ArrayList<PipelineStage>();
        if (stringCategoryFeatures != null) {
            String[] outputCols = new String[stringCategoryFeatures.length];
            Arrays.stream(stringCategoryFeatures).map(inputCol -> inputCol + INDEX_SUFFIX).collect(Collectors.toList()).toArray(outputCols);
            StringIndexer indexer = new StringIndexer()
                    .setInputCols(stringCategoryFeatures)
                    .setOutputCols(outputCols)
                    .setStringOrderType("alphabetAsc"); //"frequencyDesc"/“frequencyAsc”/“alphabetDesc”/“alphabetAsc”
            stageList.add(indexer);
            //如标签列为字符串定类数据，则需使用
            if (ArrayUtils.contains(stringCategoryFeatures, labelCol)) {
                labelCol = labelCol + INDEX_SUFFIX;
            }
            //将categoryFeatures中的字符串分类数据名称修改为带后缀的名称
            List<String> newCategoryFeatures = new ArrayList<>();
            if (categoryFeatures != null) {
                Arrays.stream(categoryFeatures).forEach(feature -> {
                    if (ArrayUtils.contains(stringCategoryFeatures, feature)) {
                        newCategoryFeatures.add(feature + INDEX_SUFFIX);
                    } else {
                        newCategoryFeatures.add(feature);
                    }
                });
                newCategoryFeatures.toArray(categoryFeatures);
            }

            List<String> newNoneCategoryFeatures = new ArrayList<>();
            if (noneCategoryFeatures != null) {
                Arrays.stream(noneCategoryFeatures).forEach(feature -> {
                    if (ArrayUtils.contains(stringCategoryFeatures, feature)) {
                        newNoneCategoryFeatures.add(feature + INDEX_SUFFIX);
                    } else {
                        newNoneCategoryFeatures.add(feature);
                    }
                });
                newNoneCategoryFeatures.toArray(noneCategoryFeatures);
            }
        }
        //非分类属性处理缺省值，支持mode,mean,median,custom
        if (noneCategoryFeatures != null) {
            String[] noneCategoryFeaturesImputed = new String[noneCategoryFeatures.length];
            Arrays.stream(noneCategoryFeatures).map(feature -> feature + IMPUTED_SUFFIX)
                    .collect(Collectors.toList())
                    .toArray(noneCategoryFeaturesImputed);
            Imputer imputer = new Imputer()
                    .setInputCols(noneCategoryFeatures)
                    .setOutputCols(noneCategoryFeaturesImputed)
                    .setStrategy("mean");//["mean", "median", "mode"]
            stageList.add(imputer);
            noneCategoryFeatures = noneCategoryFeaturesImputed;
        }

        //OneHotEncoder
        String[] categoryFeatureVectors = null;
        if (categoryFeatures != null) {
            categoryFeatureVectors = new String[categoryFeatures.length];
            Arrays.stream(categoryFeatures).map(item -> item + VECTOR_SUFFIX)
                    .collect(Collectors.toList())
                    .toArray(categoryFeatureVectors);
            OneHotEncoder encoder = new OneHotEncoder()
                    .setInputCols(categoryFeatures)
                    .setOutputCols(categoryFeatureVectors)
                    .setDropLast(true);//setHandleInvalid("keep")
            stageList.add(encoder);
        }

        //合并特征属性为单列数据 VectorAssembler
        if (!ArrayUtils.contains(trainingData.columns(),featuresCol) ||
            stageList.size()>1){
            String[] features = new String[(categoryFeatures == null ? 0 : categoryFeatures.length) +
                    (noneCategoryFeatures == null ? 0 : noneCategoryFeatures.length)];
            if (categoryFeatureVectors != null) {
                System.arraycopy(categoryFeatureVectors, 0, features, 0, categoryFeatureVectors.length);
            }
            if (noneCategoryFeatures != null) {
                System.arraycopy(noneCategoryFeatures, 0, features,
                        categoryFeatureVectors==null?0:categoryFeatureVectors.length, noneCategoryFeatures.length);
            }
            //handleInvalid:'skip' (filter out rows with invalid data), 'error' (throw an error),
            // or 'keep' (return relevant number of NaN in the output)
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(features)
                    .setOutputCol(featuresCol)
                    .setHandleInvalid("keep");
            stageList.add(assembler);
        }
        //最大值，最小值缩放
        if (scaleByMinMax){
            String outputCol = featuresCol + SCALED_SUFFIX;
            MinMaxScaler scaler = new MinMaxScaler()
                    .setInputCol(featuresCol)
                    .setOutputCol(outputCol);
            stageList.add(scaler);
            featuresCol = outputCol;
        }
        //PCA TODO 如何设置合理的K,暂时不降维，但转换为全垂直的变量
        String outputCol = featuresCol + PCA_SUFFIX;
        PCA pca = new PCA()
                .setInputCol(featuresCol)
                .setOutputCol(outputCol)
                .setK((int)total_features_count);
        featuresCol = outputCol;
        stageList.add(pca);

        PipelineStage[] stages = new PipelineStage[stageList.size()];
        stageList.toArray(stages);
        Pipeline pipeline = new Pipeline().setStages(stages);

        PipelineModel model = pipeline.fit(trainingData);
        return model;
    }

    /**
     * 根据训练的模型转换数据，训练集、测试集、预测集需要使用统一的模型转换数据
     *
     * @param source
     * @param model
     * @return
     */
    public Dataset<Row> transform(Dataset<Row> source,PipelineModel model){
        String[] columns = source.columns();
        Dataset<Row> result = source;
        if (model == null){
            return result;
        }
        if (!ArrayUtils.contains(columns, featuresCol)){
            result = model.transform(source);
        }
        //保留建模需要的所有列labelCol,featuresCol,weightCol,additionCols
        List<String> modelingCols = new ArrayList<String>();
        modelingCols.add(featuresCol);
        if (weightCol != null){
            modelingCols.add(weightCol);
        }
        if (additionCols != null){
            modelingCols.addAll(Arrays.asList(additionCols));
        }
        String[] modelCols = new String[modelingCols.size()];
        modelingCols.toArray(modelCols);
        result = result.select(labelCol,modelCols);
        return result;
    }

    /**
     * 计算各分类字段的取值数量，总和的key为total_category_count
     * @param dataset
     * @return
     */
    public Map<String,Long> countDistinctCategoryFeatures(Dataset<Row> dataset){
        Map<String,Long> countMap = new HashMap<>();
        if (categoryFeatures != null && categoryFeatures.length > 0) {
            List<Column> list = new ArrayList<>();
            Arrays.stream(categoryFeatures).forEach(column->{
                list.add(functions.countDistinct(column).name(column));
            });
            Column[] countColumns = new Column[list.size()];
            list.toArray(countColumns);
            Dataset<Row> countDataset = null;
            if (countColumns.length == 1) {
                countDataset = dataset.agg(countColumns[0]);
            } else {
                countDataset = dataset.agg(countColumns[0],ArrayUtils.subarray(countColumns, 1, countColumns.length));
            }
            Row row = countDataset.first();
            Arrays.stream(categoryFeatures).forEach(column->{
                countMap.put(column,row.getAs(column));
            });
        }
        long total_category_count = countMap.values().stream().mapToLong(count -> count).sum();
        countMap.put(TOTAL_CATEGORY_COUNT,total_category_count);
        return countMap;
    }

}
