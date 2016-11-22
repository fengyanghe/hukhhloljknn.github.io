---
layout: post
title: Drug Targets Properties Analysis and Prediction with Machine Learning
description: Yanghe Feng, Tengjiao Wang
image: assets/images/pic06.jpg
---

Motivation: Identifying new drug protein plays an important role in pharmaceutical and biomedical research. Before new drug target get manual tested and put into experimental study, a globe filter method is more efficient, practical and economical. New methods are developed to predict new drug targets, but most of them are hypothesis specific or data specific, thus, a more general computational method is required.
Result: In this study, we first category the target protein as carrier, transporter, and enzyme, gained 153 overlap proteins among these three categories which should be further analyzed. Using their sequence information we gained their <!--excerpt-->chemical and physical properties. By using the pooled drug-target data, we designed our indicator with two schemes. The first scheme focused on increasing prediction accuracy and the second one focused on shrinking the prediction scope. By integrating ada-boost and svm method, our classifier predicted 1797 and 207 potential new drug targets respectively. We randomly chose some targets from the second result, after manually confirmed online, we found out several predicted targets are already published as new drug targets.
