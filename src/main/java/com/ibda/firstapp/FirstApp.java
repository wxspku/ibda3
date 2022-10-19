package com.ibda.firstapp;

import tech.tablesaw.api.Table;

import java.io.IOException;

public class FirstApp {
    public static void main(String[] args) throws IOException {
        String clzRoot = FirstApp.class.getResource("/").getFile();
        String path = clzRoot + "/data/baseball.csv";
        Table tornadoes = Table.read().csv(path);
        System.out.println(tornadoes.columnNames());
        System.out.println(tornadoes.shape());
    }
}
