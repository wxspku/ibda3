package com.ibda.spark.recommendation;

import com.ibda.spark.SparkMLTest;
import com.ibda.spark.regression.ModelColumns;
import com.ibda.spark.regression.SparkHyperModel;
import com.ibda.util.FilePathUtil;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.io.IOException;
import java.io.Serializable;

public class ALSTest extends SparkMLTest<ALS, ALSModel> {

    public static class ItemRating implements Serializable {
        Integer userId;
        Integer itemId;
        Float rating = null;
        private Long timestamp;

        public ItemRating(Integer userId, Integer itemId, Float rating, Long timestamp) {
            this.userId = userId;
            this.itemId = itemId;
            this.rating = rating;
            this.timestamp = timestamp;
        }

        public Integer getUserId() {
            return userId;
        }

        public void setUserId(Integer userId) {
            this.userId = userId;
        }

        public Integer getItemId() {
            return itemId;
        }

        public void setItemId(Integer itemId) {
            this.itemId = itemId;
        }

        public Float getRating() {
            return rating;
        }

        public void setRating(Float rating) {
            this.rating = rating;
        }

        public Long getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(Long timestamp) {
            this.timestamp = timestamp;
        }

        public static ItemRating parseRating(String str) {
            String[] fields = str.split("::");
            if (fields.length != 4) {
                throw new IllegalArgumentException("Each line must contain 4 fields");
            }
            int userId = Integer.parseInt(fields[0]);
            int movieId = Integer.parseInt(fields[1]);
            Float rating = StringUtils.isAllBlank(fields[2]) ? null : Float.parseFloat(fields[2]);
            Long timestamp = Long.parseLong(fields[3]);
            return new ItemRating(userId, movieId, rating, timestamp);
        }
    }

    @Override
    public void initTrainingParams() {
        trainingParams.put("maxIter", 50);
        trainingParams.put("regParam", 0.2);
        trainingParams.put("userCol", "userId");
        trainingParams.put("itemCol", "itemId");
        trainingParams.put("ratingCol", "rating");
    }

    //加载test01的测试数据
    protected void loadTest01Data() {
        modelColumns = new ModelColumns(null, null, "rating");
        modelColumns.setAdditionCols(new String[]{"userId", "itemId"});
        JavaRDD<ItemRating> ratingsRDD = spark
                .read().textFile("data/mllib/als/sample_movielens_ratings.txt").javaRDD()
                .map(ItemRating::parseRating);
        Dataset<Row> ratings = spark.createDataFrame(ratingsRDD, ItemRating.class);
        ratings.show();
        splitDataset(ratings);

    }

    protected void loadTest02Data() {

    }

    @Override
    public void test01LearningEvaluatingPredicting() throws IOException {
        super.test01LearningEvaluatingPredicting();
        String modelPath = FilePathUtil.getAbsolutePath("output/" + modelClass.getSimpleName() + ".model", true);
        SparkHyperModel<ALSModel> loadedModel = SparkHyperModel.loadFromModelFile(modelPath, modelClass);
        ALSModel model = loadedModel.getModel();

        System.out.println("Generate top 10 movie recommendations for each user");
        Dataset<Row> userRecs = model.recommendForAllUsers(10);
        userRecs.show();

        System.out.println("Generate top 10 user recommendations for each movie");
        Dataset<Row> movieRecs = model.recommendForAllItems(10);
        movieRecs.show();

        System.out.println("Generate top 10 movie recommendations for a specified set of users");
        Dataset<Row> users = datasets[0].select(model.getUserCol()).distinct().limit(3);
        Dataset<Row> userSubsetRecs = model.recommendForUserSubset(users, 10);
        userSubsetRecs.show();

        System.out.println("Generate top 10 user recommendations for a specified set of movies");
        Dataset<Row> movies = datasets[0].select(model.getItemCol()).distinct().limit(3);
        Dataset<Row> movieSubSetRecs = model.recommendForItemSubset(movies, 10);
        movieSubSetRecs.show();
    }

    @Override
    public void test02MachineLearning() throws IOException {
        loadTest02Data();
        test01LearningEvaluatingPredicting();
    }
}
