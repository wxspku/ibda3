package com.ibda.spark.regression;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.*;
import java.util.stream.Collectors;

/**
 * TODO 需考虑支持StringIndex转换，转换后形成新的Category变量，可能作为features中的一列，也可能成为label列
 */
public class ModelColumns {

    private static final String REGRESSION_FEATURES_VECTOR = "regression_features_vector";
    private static final String REGRESSION_FEATURES_DEFAULT = "features";
    private static final String VECTOR_SUFFIX = "_vector";
    private static final String INDEX_SUFFIX = "_index";

    public static ModelColumns MODEL_COLUMNS_DEFAULT =  new ModelColumns();

    String[] noneCategoryFeatures ; //特征属性列表
    String[] categoryFeatures;   //特征属性中的分类属性，需要进行OneHotEncoder处理
    String[] stringCategoryFeatures ;  //使用字符串类型的分类属性，需要转换为其索引号，按照"frequencyDesc"方式编码
    String featuresCol = REGRESSION_FEATURES_DEFAULT;
    String labelCol = "label";         //观察结果列
    String predictCol = "prediction";       //预测结果列
    String probabilityCol = "probability";   //或然列，逻辑回归时的概率值

    /**
     * 使用缺省值
     */
    private ModelColumns(){
        super();
    }

    /**
     *
     * @param noneCategoryFeatures
     * @param categoryFeatures
     * @param labelCol
     */
    public ModelColumns(String[] noneCategoryFeatures, String[] categoryFeatures, String labelCol){
        this.noneCategoryFeatures = noneCategoryFeatures;
        this.categoryFeatures = categoryFeatures;
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
        this.noneCategoryFeatures = noneCategoryFeatures;
        this.categoryFeatures = categoryFeatures;
        this.labelCol = labelCol;
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

    @Override
    public String toString() {
        return "ModelColumns{" +
                "noneCategoryFeatures=" + Arrays.toString(noneCategoryFeatures) +
                ", categoryFeatures=" + Arrays.toString(categoryFeatures) +
                ", featuresCol='" + featuresCol + '\'' +
                ", labelCol='" + labelCol + '\'' +
                ", predictCol='" + predictCol + '\'' +
                ", probabilityCol='" + probabilityCol + '\'' +
                '}';
    }

    /**
     * 根据数据集训练PipelineModel
     * @param trainingData
     * @return
     */
    public PipelineModel getPipelineModel(Dataset<Row> trainingData) {
        initPipeline();
        PipelineModel model = pipeline.fit(trainingData);
        return model;
    }

    /**
     * 根据训练的模型转换数据，训练集、测试集、预测集需要使用统一的模型转换数据
     * @param model
     * @param source
     * @return
     */
    public Dataset<Row> transform(PipelineModel model,Dataset<Row> source){
        return model.transform(source).select(labelCol,featuresCol);
    }

    Pipeline pipeline = null;

    /**
     * 初始化pipeline
     */
    private void initPipeline(){
        if (pipeline != null){
            return;
        }
        List<PipelineStage> stageList = new ArrayList<PipelineStage>();
        //StringIndexer的fit样本自动编码 frequencyDesc，TODO 如果需要外部的编码表，需要在外面自行处理
        if (stringCategoryFeatures != null){
            String[] outputCols = new String[stringCategoryFeatures.length];
            Arrays.stream(stringCategoryFeatures).map(inputCol->inputCol + INDEX_SUFFIX).collect(Collectors.toList()).toArray(outputCols);
            StringIndexer indexer = new StringIndexer()
                    .setInputCols(stringCategoryFeatures)
                    .setOutputCols(outputCols)
                    .setStringOrderType("frequencyDesc"); //"frequencyDesc"/“frequencyAsc”/“alphabetDesc”/“alphabetAsc”
            stageList.add(indexer);
            //如标签列为字符串定类数据，则需使用
            if (ArrayUtils.contains(stringCategoryFeatures,labelCol)){
                labelCol = labelCol + INDEX_SUFFIX;
            }
            //将categoryFeatures中的字符串分类数据名称修改为带后缀的名称
            List<String> newCategoryFeatures = new ArrayList<>();
            if (categoryFeatures != null){
                Arrays.stream(categoryFeatures).forEach(categoryFeature ->{
                    if (ArrayUtils.contains(stringCategoryFeatures,categoryFeature)){
                        newCategoryFeatures.add(categoryFeature + INDEX_SUFFIX);
                    }
                    else{
                        newCategoryFeatures.add(categoryFeature);
                    }
                });
                newCategoryFeatures.toArray(categoryFeatures);
            }
        }
        //OneHotEncoder
        String[] categoryFeatureVectors = null;
        if (categoryFeatures != null){
            categoryFeatureVectors = new String[categoryFeatures.length];
            Arrays.stream(categoryFeatures).map(item -> item + VECTOR_SUFFIX)
                    .collect(Collectors.toList())
                    .toArray(categoryFeatureVectors);
            OneHotEncoder encoder = new OneHotEncoder()
                    .setInputCols(categoryFeatures)
                    .setOutputCols(categoryFeatureVectors)
                    .setHandleInvalid("error");
            stageList.add(encoder);
        }
        //合并特征属性为单列数据 VectorAssembler
        String[] features = new String[(categoryFeatures == null ? 0 : categoryFeatures.length) +
                (noneCategoryFeatures == null ? 0 : noneCategoryFeatures.length)];
        if (categoryFeatureVectors != null) {
            System.arraycopy(categoryFeatureVectors, 0, features, 0, categoryFeatureVectors.length);
        }
        if (noneCategoryFeatures != null) {
            System.arraycopy(noneCategoryFeatures, 0, features, categoryFeatureVectors.length, noneCategoryFeatures.length);
        }
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(features)//new String[]{"score"}
                .setOutputCol(featuresCol);//"score_vector"
        stageList.add(assembler);
        PipelineStage[] stages = new PipelineStage[stageList.size()];
        stageList.toArray(stages);
        pipeline = new Pipeline().setStages(stages);
    }
}
