# Learning from Demonstration with CNMPs

This repository contains the implementation of Homework 4 for CMPE591, which focuses on learning from demonstration using Conditional Neural Processes (CNPs). The goal is to train a CNP model to predict robot end-effector and object positions based on partial demonstrations and object height.

## Overview

In this homework, we collect demonstrations that consist of (t, e_y, e_z, o_y, o_z, h) where:
- t is the time (query dimension)
- e_y, e_z are the end-effector cartesian coordinates
- o_y, o_z are the object cartesian coordinates
- h is the height of the object (condition)

The robot randomly moves its end-effector in the y-z plane, sometimes hitting the object and sometimes not. The height of the object is random and is provided from the environment.

We train a CNP model to predict the end-effector and object positions given:
1. The time t (query dimension)
2. The object height h (condition)
3. Context points consisting of other (t, e_y, e_z, o_y, o_z) points

## Results

The test script produces:
1. MSE statistics for both end-effector and object predictions
2. Error bar plots comparing the MSE for end-effector and object predictions
3. Visualization of predictions vs. ground truth for selected test cases

![loss_curves_20250420_062032](https://github.com/user-attachments/assets/360aad0d-50bc-4fbb-bd6b-93e8737bf574)


![error_bars](https://github.com/user-attachments/assets/b786bf0e-bbbc-4772-bbf4-dd2f44128228)


![predictions](https://github.com/user-attachments/assets/72661723-67f9-4a75-a0b7-6a7a1c75d186)
