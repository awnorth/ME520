% ME 520 Machine Learning Assignment
% By Andrew North - Due 11/18/2021

clc
clear 

load("TextMSGDataSet.mat")
% Struct Messages: Messages.label Messages.msg

% % ---------- Histogram of all data ---------- %
% A = struct2table(Messages);
% B = A.label;
% C = categorical(B,[1 0],{'spam (13.4%)','nonspam (86.6%)'});
% histogram(C);
% title("All Samples")
% ylabel("Samples")


% Create spam and non spam vectors
spam = [""];
nonspam = [""];
for n = 1:length(Messages)
    class = Messages(n).label;
    if class == 1
        spam(end+1,:) = Messages(n).msg;
    else
        nonspam(end+1,:) = Messages(n).msg;
    end
end
% correct spam and nonspam vector lengths. Remove first element
spam = spam(2:end);
nonspam = nonspam(2:end);

% OLD
% ------------------------ 70/30 training/test ------------------------ %
spam_indices = randperm(length(spam));
train_spam = spam(spam_indices(1:520),:); % Take 520 random spam
test_spam = spam(spam_indices(521:end),:); % Take remaining 227

nonspam_indices = randperm(length(nonspam));
train_nonspam = nonspam(nonspam_indices(1:3380),:); % Take 3380 random nonspam
test_nonspam = nonspam(nonspam_indices(3381:end),:); % Take remaining 1447

% Take 70% of data randomly for training
% Take 30% of data randomly for testing
indices = randperm(length(Messages));
Train = Messages(indices(1:3900));
Test = Messages(indices(3901:end)); 

% ---------------------- 10-Fold Cross Validation ---------------------- %
for n = 1:10
    
end


% --------------- Continuous Feature - CAPITAL Letters --------------- %
spam_cap_sum = 0;
spam_cap = [];
for n = 1:length(spam)
    spam_cap(n) = sum(isstrprop(spam(n),'upper'));
    spam_cap_sum = spam_cap_sum + sum(isstrprop(spam(n),'upper'));
    spam_cap_avg = spam_cap_sum/length(spam);
end

nonspam_cap_sum = 0;
non_spam = [];
for n = 1:length(nonspam)
    nonspam_cap(n) = sum(isstrprop(nonspam(n),'upper'));
    nonspam_cap_sum = nonspam_cap_sum + sum(isstrprop(nonspam(n),'upper')); 
    nonspam_cap_avg = nonspam_cap_sum/length(nonspam);
end

cap = [];
for n = 1:length(Train)
    cap(n) = sum(isstrprop(Train(n).msg,'upper'));
end

cap_test = [];
for n = 1:length(Test)
    cap_test(n) = sum(isstrprop(Test(n).msg,'upper'));
end

%cap_all
cap_all = [];
for n = 1:length(Messages)
    cap_all(n) = sum(isstrprop(Messages(n).msg,'upper'));
end

% --------------- Continuous Feature - String Lengths --------------- %
Len = 0;
for n = 1:length(spam)
    Lengths(n) = strlength(spam(n));
    Len = Len + Lengths(n);
%     total_spam_len = total_spam_len + strlength(spam(n));
    avg_spam_len = Len/length(spam);
end

total_nonspam_len = 0;
for n = 1:length(nonspam)
    nonspam_lengths(n) = strlength(nonspam(n));
    total_nonspam_len = total_nonspam_len + nonspam_lengths(n);
%     total_nonspam_len = total_nonspam_len + strlength(nonspam(n));
    avg_nonspam_len = total_nonspam_len/length(nonspam);
end

% len_all
for n = 1:length(Messages)
    len_all(n) = strlength(Messages(n).msg);
end

scale = length(nonspam)/length(spam);

x_nonspam = [1:length(nonspam)];
scatter(x_nonspam,nonspam_lengths,7,'filled')
hold on

x_spam = [1:scale:scale*length(spam)];
scatter(x_spam,Lengths,7,'filled')

legend('non-spam','spam')
title("Data String Lengths")
xlabel("sample");
ylabel("sample length");
hold off

% ------- Continuous Feature - qty integers 1 2 3 4 5 6 7 8 9  ------- %
i = ['0','1','2','3','4','5','6','7','8','9'];
for n = 1:length(spam)
    spam_int_sum = 0;
%     s1(n) = count(spam(n),"1");
%     s2(n) = count(spam(n),"2");
%     s3(n) = count(spam(n),"3");
    for m = 1:10
        s(m) = count(spam(n),i(m));
        spam_int_sum = spam_int_sum + s(m);
        spam_int(n) = spam_int_sum;
    end
end
spam_int_avg = sum(spam_int_sum)/length(spam);

for n = 1:length(nonspam)
    nonspam_int_sum = 0;
    for m = 1:10
        s(m) = count(nonspam(n),i(m));
        nonspam_int_sum = nonspam_int_sum + s(m);
        nonspam_int(n) = nonspam_int_sum;
    end
end
nonspam_int_avg = sum(nonspam_int_sum)/length(nonspam);

% now implement number of integers into the Training and Testing data
for n = 1:length(Train)
      train_int(n) = 0;
    for m = 1:10
        train_int(n) = train_int(n) + count(Train(n).msg,i(m));
    end
end

for n = 1:length(Test)
      test_int(n) = 0;
    for m = 1:10
        test_int(n) = test_int(n) + count(Test(n).msg,i(m));
    end
end

% int_all
for n = 1:length(Messages)
      int_all(n) = 0;
    for m = 1:10
        int_all(n) = int_all(n) + count(Messages(n).msg,i(m));
    end
end

% ---------- Continuous Feature - qty "call" & "txt" ---------- %
feature04 = count(extractfield(Messages,"msg")," call ","IgnoreCase",true)'+count(extractfield(Messages,"msg")," txt ","IgnoreCase",true)';
feature05 = count(extractfield(Messages,"msg"),"win","IgnoreCase",true)'+count(extractfield(Messages,"msg"),"free","IgnoreCase",true)'+count(extractfield(Messages,"msg"),"prize","IgnoreCase",true)';
feature06 = count(extractfield(Messages,"msg"),"claim","IgnoreCase",true)';
feature07 = count(extractfield(Messages,"msg"),"urgent","IgnoreCase",true)';
feature08 = count(extractfield(Messages,"msg"),"guaranteed","IgnoreCase",true)';
feature09 = count(extractfield(Messages,"msg"),"cash","IgnoreCase",true)';
feature10 = count(extractfield(Messages,"msg"),"www","IgnoreCase",true)';

% ---------- Create Label for Feature Table ---------- %
label_all = extractfield(Messages,"label");

% Compute Features for All Data
features = [len_all' cap_all' int_all' feature04 feature05 feature06 feature07 feature08 feature09 feature10 label_all']
names = {'len','cap','int','call_txt','win_free_prize','claim','urgent','guaranteed','cash','www','label'};
feature_table = array2table(features,'VariableNames',names);

% Boxplot of len_all
boxplot(feature_table.int,feature_table.label,'labels',{'non-spam','spam'})
title("Quantity of Integers by Class")
ylabel("Qty of Integers")

% ------------------------ 70/30 training/test ------------------------ % %

indices = randperm(height(feature_table));
train_table = feature_table(indices(1:3900),:); % Take 3900 random samples
test_table = feature_table(indices(3901:end),:); % take remaining 1674 samples

% ---------- Create continuous freature structure ---------- %
for n = 1:length(Train)
    Lengths(n) = strlength(Train(n).msg);
end

for n = 1:length(Test)
    Lengths_test(n) = strlength(Test(n).msg);
end

% Creating the Training Struct
Length = cell2struct(num2cell(Lengths),'lengths');
Cap = cell2struct(num2cell(cap),'capitals');
feat_struct = struct('capitals',{Cap.capitals},'lengths',{Length.lengths},'class_label', {Train.label});
features_continuous = struct2table(feat_struct);

% Creating the Testing Struct
Length_test = cell2struct(num2cell(Lengths_test),'lengths');
Cap_test = cell2struct(num2cell(cap_test),'capitals');
feat_struct_test = struct('capitals',{Cap_test.capitals},'lengths',{Length_test.lengths},'class_label', {Test.label});
features_continuous_test = struct2table(feat_struct_test);

% gscatter(features_continuous.capitals,features_continuous.lengths,features_continuous.class_label)
% title("Quantity of Capital Letters vs. Length of all samples")
% xlabel("Quantity of Capital Letters")
% ylabel("Length")
% legend('non-spam','spam')


gscatter(features_continuous.lengths,features_continuous.capitals,features_continuous.class_label)
title("Length vs. Quantity of Capital Letters for all samples")
ylabel("Quantity of Capital Letters")
xlabel("Length")
legend('non-spam','spam')



% knnmodel = fitcknn(features_continuous,"class_label");
% 
% predicted = predict(knnmodel,features_continuous_test);
% 
% iscorrect = predicted == features_continuous_test.class_label;
% accuracy = sum(iscorrect)/numel(predicted)
% confusionchart(features_continuous_test.class_label,predicted)


% ------------- Selecting and Computing a Set of Features ------------- %
% IGNORE = true; % ignore case while searching strings
% 
% P_spam = length(spam)/length(Messages); % Probability of class spam = Prior
% P_notspam = length(nonspam)/length(Messages); % Probability of class nonspam = Prior
% 
% % Feature List
% feature1 = "urgent";  % Does the msg contain "free"
% feature2 = "call";  % Does the msg contain "call"
% features12 = {'claim','urgent'}; % Lets assume freature 1 & 2 are dependent

% ---------------------------- feature 1 ---------------------------- %
% feature1 = "guaranteed";
% IGNORE = true;
% % P(feature1|class spam)   P("free"|spam)
% TF1 = contains(spam,feature1,'IgnoreCase',IGNORE);
% P_feature1_spam = sum(TF1)/length(TF1)*100;
% 
% % P(feature1|class nonspam)   P("free"|nonspam)
% TFF = contains(nonspam,feature1,'IgnoreCase',IGNORE);
% P_feature1_notspam = sum(TFF)/length(TFF)*100;

% --------------------------- feature 1 & 2 --------------------------- %
% P('free','call'|spam) = Likelihood
% P_12_spam = contains(train_spam,features12,'IgnoreCase',IGNORE);
% P_12_spam = sum(P_12_spam)/length(P_12_spam);
% 
% % P('free','call'|nonspam) = likelihood
% P_12_nonspam = contains(train_nonspam,features12,'IgnoreCase',IGNORE);
% P_12_nonspam = sum(P_12_nonspam)/length(P_12_nonspam);
% 
% 
% % P(spam|'free','call')
% P_spam_12 = (P_12_spam*P_spam)/(P_12_spam+P_12_nonspam);
% % P(nonspam|'free','call')
% P_nonspam_12 = (P_12_nonspam*P_notspam)/(P_12_spam+P_12_nonspam);


% if P_spam_12 > P_nonspam_12
%     spam = 1;
% else
%     spam = 0;
% end
% 
% spam

% P_spam_12 > P_nonspam_12
% So given features 1 & 2, the class is spam

% --------------- Create the kNN model using "Train" data --------------- %

% Combined features directly without normalizing  
% Feature1 = cell2struct(num2cell(count(extractfield(Train,"msg"),"call","IgnoreCase",true)),"feature1");
% Feature2 = cell2struct(num2cell(count(extractfield(Train,"msg"),"free","IgnoreCase",true)),"feature2");
% Feature3 = cell2struct(num2cell(count(extractfield(Train,"msg"),"claim","IgnoreCase",true)),"feature3");
% Feature4 = cell2struct(num2cell(count(extractfield(Train,"msg"),"win","IgnoreCase",true)),"feature4");
% Features = struct('feature1',{Feature1.feature1},'feature2',{Feature2.feature2},'feature3',{Feature3.feature3},'feature4',{Feature4.feature4},"label",{Label.label});

feature1 = count(extractfield(Train,"msg"),"call","IgnoreCase",true)';
feature2 = count(extractfield(Train,"msg"),"free","IgnoreCase",true)';
feature3 = count(extractfield(Train,"msg"),"claim","IgnoreCase",true)';
feature4 = count(extractfield(Train,"msg"),"win","IgnoreCase",true)';
feature5 = count(extractfield(Train,"msg"),"prize","IgnoreCase",true)';
feature6 = count(extractfield(Train,"msg"),"cash","IgnoreCase",true)';

% Normalize length. New values 0 - 0.9978
L = (Lengths - min(Lengths))/max(Lengths);
Length = cell2struct(num2cell(L),'lengths');

% normalize CAPITALS. New values 0 - 1 
C = (cap - min(cap))/max(cap);
Capital = cell2struct(num2cell(C),'capitals');

% normalize CAPTIALS. # capitals / length
C_L = cap./Lengths;
Capitals_Length = cell2struct(num2cell(C_L),'capitals_length');

% normalize number of integers 0 1 2 3 4 5 ...
I = train_int/max(train_int);
Integers = cell2struct(num2cell(I),'integers');

Label = cell2struct(num2cell(extractfield(Train,"label")),"label");


for n = 1:length(Train)
    features(n) = feature1(n)+feature2(n)+feature3(n)+feature4(n)+feature5(n)+feature6(n);
    f(n) = sum(features(n))/6;
end

f = cell2struct(num2cell(f),"features");
Features = struct('feature',{f.features},'length',{Length.lengths},'capitals',{Capital.capitals},'capitals_length',{Capitals_Length.capitals_length},'integers',{Integers.integers},'label',{Label.label});
% Features = struct('feature',{f.features},'length',{Length.lengths},'capitals',{Capital.capitals},'label',{Label.label});

Features = struct2table(Features);

gscatter(Features.length,Features.capitals,Features.label)

% ----------------------------- Step 2  ----------------------------- %
% For Loop to find best k value parameter
% k = 5;
% val_acc = [];
% mod_acc = [];
% 
% for k = 1:15
%     cvpt = cvpartition(train_table.label,"KFold",10) % 10 Fold Cross Validation
%     knnmodel = fitcknn(train_table,"label","CVPartition",cvpt,'NumNeighbors',k)
%     
%     test_table2 = removevars(test_table,{'label'}); % remove 'label' column- Don't think I need this.
%     
%     mdlLoss = kfoldLoss(knnmodel)
%     validation_accuracy = 100 - sqrt(mdlLoss*100) 
%     
%     predicted = predict(knnmodel.Trained{1,1},test_table);
%     iscorrect = predicted == test_table.label;
%     accuracy = sum(iscorrect)/numel(predicted);
%     accuracy_p = accuracy*100
% %     val_acc((k+1)/2) = validation_accuracy; % for a large range of k
% %     mod_acc((k+1)/2) = accuracy_p;
%     val_acc(k) = validation_accuracy;
%     mod_acc(k) = accuracy_p;
% end

% x = [1:15];
% plot(x,val_acc)
% hold on
% plot(x,mod_acc)
% title("K-Value vs. Accuracy")
% xlabel("K-Value in Nearest Neighbor Classifier")
% ylabel("Accuracy")
% legend('Validation Accuracy','Model Accuracy')


% ----------------------------- Step 3  ----------------------------- %

k = 5; % Choosing 5 becasue this provides best accuracy in the model.

cvpt = cvpartition(train_table.label,"KFold",10) % 10 Fold Cross Validation

knnmodel = fitcknn(train_table,"label","CVPartition",cvpt,'NumNeighbors',k)
% knnmodel = fitcknn(Features,"label","NumNeighbors",5) % HERE IS THE MODEL!

% Test the new model:
test_table2 = removevars(test_table,{'label'}); % remove 'label' column- Don't think I need this.

% mdlLoss = kfoldLoss(knnmodel,'Mode','individual')  % Looks at mean
% squared error for each fold
% boxchart(mdlLoss)
mdlLoss = kfoldLoss(knnmodel)
validation_accuracy = 100 - sqrt(mdlLoss*100) 


predicted = predict(knnmodel.Trained{1,1},test_table);

iscorrect = predicted == test_table.label;
accuracy = sum(iscorrect)/numel(predicted);
accuracy_p = accuracy*100

val_acc(k) = validation_accuracy;
mod_acc(k) = accuracy_p;

confusionchart(test_table.label,predicted)
title("Confusion Matrix with Raw Unnormalized Data")



% -------------- Find the "Test" data features 1,2,3,etc. -------------- %

feature1_test = count(extractfield(Test,"msg"),"call","IgnoreCase",true)';
feature2_test = count(extractfield(Test,"msg"),"free","IgnoreCase",true)';
feature3_test = count(extractfield(Test,"msg"),"claim","IgnoreCase",true)';
feature4_test = count(extractfield(Test,"msg"),"win","IgnoreCase",true)';
feature5_test = count(extractfield(Test,"msg"),"prize","IgnoreCase",true)';
feature6_test = count(extractfield(Test,"msg"),"urgent","IgnoreCase",true)';

% Normalize length. New values 0 - 0.9978
L_test = (Lengths_test - min(Lengths_test))/max(Lengths_test);
Length_test = cell2struct(num2cell(L_test),'lengths');

% normalize CAPITALS. New values 0 - 1 
C_test = (cap_test - min(cap_test))/max(cap_test);
Capital_test = cell2struct(num2cell(C_test),'capitals');

% normalize CAPTIALS. # capitals / length
C_L_test = cap_test./Lengths_test;
Capitals_Length_test = cell2struct(num2cell(C_L_test),'capitals_length');

% normalize number of integers 0 1 2 3 4 5 ...
I_test = test_int/max(test_int);
Integers_test = cell2struct(num2cell(I_test),'integers');


Label_test = cell2struct(num2cell(extractfield(Test,"label")),"label");

for n = 1:length(Test)
    features_test(n) = feature1_test(n)+feature2_test(n)+feature3_test(n)+feature4_test(n)+feature5_test(n)+feature6_test(n);
    f_test(n) = sum(features_test(n))/6;
end

f_test = cell2struct(num2cell(f_test),"features");
Features_test = struct('feature',{f_test.features},'length',{Length_test.lengths},'capitals',{Capital_test.capitals},'capitals_length',{Capitals_Length_test.capitals_length},'integers',{Integers_test.integers},'label',{Label_test.label});
% Features_test = struct('feature',{f_test.features},'length',{Length_test.lengths},'capitals',{Capital_test.capitals},'label',{Label_test.label});

Features_test = struct2table(Features_test);



% predicted = predict(knnmodel,feature_table)
% test_labels = extractfield(feature_table,"label");
% iscorrect = predicted == test_labels';
% accuracy = sum(iscorrect)/numel(iscorrect);  % Gives about 94% accuracy
% confusionchart(test_labels,predicted)

% predicted = predict(knnmodel,Features_test);
% test_labels = extractfield(Test,"label");
% iscorrect = predicted == test_labels';
% accuracy = sum(iscorrect)/numel(iscorrect);  % Gives about 94% accuracy
% confusionchart(test_labels,predicted)





% gscatter(Features.capitals_length,Features.feature,Features.label,'br','xo')
% title("Capitals/Length vs. Combined Word Features")
% ylabel("Quantity Specific Words")
% xlabel("Capitals/Length")
% legend('non-spam','spam')

% gscatter(Features.feature,Features.capitals_length,Features.label,'br','xo')
% title("Combined Word Features vs.Capitals/Length")
% xlabel("Quantity Specific Words")
% ylabel("Capitals/Length")
% legend('non-spam','spam')

% gscatter(Features.integers,Features.capitals_length,Features.label,'br','xo')
% title("Qty Integers vs.Capitals/Length")
% xlabel("Qty of Integers")
% ylabel("Capitals/Length")
% legend('non-spam','spam')



% indices = crossvalind('kfold',feature_table,10)

