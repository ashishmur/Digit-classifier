//
//  main.cpp
//  ANN 3
//
//  Created by Ashish Muralidharan on 19/06/15.
//  Copyright (c) 2015 Ashish Muralidharan. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <random>
#include <math.h>
#include <stdio.h>

using namespace std;
typedef unsigned char BYTE;
long getFileSize(FILE *file)
{
    long lCurPos, lEndPos;
    lCurPos = ftell(file);
    fseek(file, 0, 2);
    lEndPos = ftell(file);
    fseek(file, lCurPos, 0);
    return lEndPos;
}
int data[60000][28][28];
int label[60000];
int temp=10;
float weights[10][784];

void random_normal(float (&weights)[10][784]){
    default_random_engine generator;
    normal_distribution<float> random_weight(0.0,1.0);
    for (int i=0; i<10; i++) {
        for (int j=0; j<784; j++) {
            weights[i][j]=random_weight(generator);
        }
    }
}

void random_normal_bias(float (&bias)[10]){
    default_random_engine generator;
    normal_distribution<float> random_weight(0.0,1.0);
    for (int i=0; i<10; i++) {
        bias[i]=random_weight(generator);
    }
}

void testing(float (&weights)[10][784],float (&bias)[10]){
    float sigma;
    float flag=0;
    
    const char *filePath = "data7";
    BYTE *fileBuf;
    FILE *file = NULL;
    if ((file = fopen(filePath, "rb")) == NULL)
        cout << "Could not open image file" << endl;
    else
        cout << "Image File opened successfully" << endl;
    long fileSize = getFileSize(file);
    fileBuf = new BYTE[fileSize];
    fread(fileBuf, fileSize, 1, file);
    printf("\n Data Succesfully transferred to filebuf \n");
    printf("filesize: %ld \n" ,fileSize);
    float sum;
    for (int i=0; i<1000; i++) {
        sum=0;
        for (int k=i*784; k<(i+1)*784; k++) {
            sum+=weights[5][k-(i*784)];
        }
        sum=sum-bias[5];
        sigma=(1/(1+(exp(-sum))));
        if (sigma>=0.5) {
            flag++;
        }
    }
    flag=flag/1000;
    printf("The accuracy is %f :",flag);
    fclose(file);
}

int main() {
    const char *filePath = "train_images";
    BYTE *fileBuf;
    FILE *file = NULL;
    if ((file = fopen(filePath, "rb")) == NULL)
        cout << "Could not open image file" << endl;
    else
        cout << "Image File opened successfully" << endl;
    long fileSize = getFileSize(file);
    fileBuf = new BYTE[fileSize];
    fread(fileBuf, fileSize, 1, file);
    printf("\n Data Succesfully transferred to filebuf \n");
    printf("filesize: %ld \n" ,fileSize);
    fclose(file);
    
    const char *filepath_label = "train_labels";
    BYTE *fileBuf_labels;
    FILE *file_label=NULL;
    if ((file_label = fopen(filepath_label, "rb")) == NULL)
        cout << "Could not open label file" << endl;
    else
        cout << "Label File opened successfully" << endl;
    long fileSize_label = getFileSize(file_label);
    fileBuf_labels = new BYTE[fileSize_label];
    fread(fileBuf_labels, fileSize_label, 1, file_label);
    printf("\n Data Succesfully transferred to filebuf_label \n");
    printf("filesize: %ld \n" ,fileSize_label);
    
    printf("\n beggining to save data in data \n");
    int k=0;
    int a=0;
    int b=0;
    for (long i=16; i<47040016; i++) {
        data[k][a][b]=fileBuf[i];
        b++;
        if (b==28) {
            b=0;
            a++;
            if (a==28) {
                a=0;
                k++;
            }
        }
    }
    printf("\n completed saving data in data");
    
    cout<<"Beggining to save labels."<<endl;
    for (long i=8; i<60008; i++) {
        label[i-8]=fileBuf_labels[i];
    }
    cout<<"Completed saving labels in data"<<endl;
    fclose(file);   // Almost forgot this
    delete[]fileBuf;
    cin.get();
    delete[]fileBuf_labels;

    cout<<"Beginning to start neural network"<<endl;
    float bias[10];
    random_normal(weights);
    random_normal_bias(bias);
    
    float sum[10];
    float error_cost[10];
    float error[10];
    float lrate=0.5;
    float delta[10];
    float signum[10];
    float weight_change[784];
    float total_error=0;
    
    for (int k=0; k<60000; k++) {
        temp=label[k];
        
        for (int i=0; i<10; i++) {
            sum[i]=0;
            for (int j=0; j<784; j++) {
                sum[i]=sum[i]+(weights[i][j]*data[k][j/28][j%28]);
            }
            sum[i]=sum[i]-bias[i];
            signum[i]=(1/(1+(exp(-sum[i]))));
            if (temp==i) {
                error_cost[i]=0.5*(pow((0.9999-signum[i]), 2));
                error[i]=0.9999-signum[i];
            }
            else {
                error_cost[i]=0.5*(pow((0.0001-signum[i]), 2));
                error[i]=0.0001-signum[i];
            }
            delta[i]=error[i]*signum[i]*(1-signum[i]);
            for (int j=0; j<784; j++) {
                weight_change[j]=delta[i]*lrate*data[k][j/28][j%28];
                weights[i][j]+=weight_change[j];
            }
            bias[i]=bias[i]+(delta[i]*lrate);
        }
        total_error=0;
        for (int i=0; i<10; i++) {
            total_error=total_error+error_cost[i];
        }
        printf("The Error at iteration %d is: %f \n\n",k,total_error);
    }
    
    testing(weights, bias);
    
    return 0;
}
