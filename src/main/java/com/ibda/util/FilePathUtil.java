package com.ibda.util;

import java.io.File;
import java.io.IOException;

public class FilePathUtil {
    /**
     *
     * @return
     */
    public static String getClassRoot(){
        return FilePathUtil.class.getResource("/").getFile();
    }

    /**
     * 当前工作路径
     * @return
     */
    public static String getWorkingDirectory() {
        try {
            return new File(".").getCanonicalPath() + "/";
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * 获取绝对路径
     * @param relativePath
     * @param relativeToWorkingDirectory true表示相对当前目录，false表示相对class root
     * @return
     */
    public static String getAbsolutePath(String relativePath,boolean relativeToWorkingDirectory){
        String basePath = relativeToWorkingDirectory?getWorkingDirectory():getClassRoot();
        basePath = basePath.replaceAll("\\\\","/");
        relativePath = relativePath.replaceAll("\\\\","/");
        if (relativePath.startsWith("/")){
            relativePath = relativePath.substring(1);
        }
        return basePath + relativePath;
    }
}
