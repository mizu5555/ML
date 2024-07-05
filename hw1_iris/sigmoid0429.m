load IRIS_IN.csv;
load IRIS_OUT.csv; 

input=IRIS_IN;
target=IRIS_OUT;
NEWtarget = zeros(150, 3);

for i = 1:size(target, 1)
    if target(i) == 1
        NEWtarget(i, :) = [1, 0, 0];
    elseif target(i) == 2
        NEWtarget(i, :) = [0, 1, 0];
    elseif target(i) == 3
        NEWtarget(i, :) = [0, 0, 1];
    end
end

training_input = input(1:75, :);
testing_input = input(76:150, :);
training_target = NEWtarget(1:75, :);
testing_target = NEWtarget(76:150, :);

final = zeros(size(training_input, 1), 3);

Whid = rand(4, 12)*2-1;
BIAShid = rand(1, 12)*2-1;
Wout = rand(12, 3)*2-1;
BIASout = rand(1, 3)*2-1;

alpha = 0.1;
EPOCH = 100;

rmse_values = zeros(EPOCH, 1); 


for epoch = 1:EPOCH
    mse = 0;

    %訓練
    for i = 1:size(training_input,1)
        input = training_input(i,:);
        target = training_target(i,:);
        
        SUMhid = input * Whid + BIAShid;
        Ahid = logsig(SUMhid);
        
        SUMout = Ahid * Wout + BIASout;
        Aout = softmax_YN(SUMout);     % purelin改softmax
        
        error = target - Aout;         %
        DELTAout = error; 
        DELTAhid = DELTAout*Wout.';

        Wout = Wout + Ahid.'*DELTAout*alpha;
        Whid = Whid + input.'*DELTAhid.*dlogsig(SUMhid,Ahid)*alpha;

    end
    
    %測試
    correct = 0;
    for i=1:size(testing_input,1)
        input = testing_input(i,:);
 
        SUMhid = input * Whid + BIAShid;
        Ahid = logsig(SUMhid);
        
        SUMout = Ahid * Wout + BIASout;
        Aout = softmax_YN(SUMout);
        final(i, :) = Aout;

        [max_value1, max_index1] = max(Aout);
        [max_value2, max_index2] = max(testing_target(i,:));
        if(max_index1==max_index2)
            correct = correct+1;
        end
    end
    rmse = sqrt(mse/75);
    fprintf("Epoch %d - RMSE: %f, ACC: %f%%\n", epoch, rmse, correct*100/75);
    
    rmse_values(epoch, :) = rmse; 
    
end

