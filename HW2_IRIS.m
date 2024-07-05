%{
CE8009 -圖形識別實務與應用
土木4B 109302545 莊明儒
%}

function HW2_IRIS_109302545

% Load data
load IRIS_IN.csv;
load IRIS_OUT.csv;
input = IRIS_IN;
target = IRIS_OUT;
RMSE_table = table('Size', [100, 3], 'VariableTypes', {'double', 'double', 'double'}, 'VariableNames', {'RMSE_1', 'RMSE_2', 'RMSE_3'});

% one-hot encoding
target_onehot = zeros(size(target, 1), 3);
for i = 1:size(target, 1)
    if target(i) == 1
        target_onehot(i, :) = [1, 0, 0];
    elseif target(i) == 2
        target_onehot(i, :) = [0, 1, 0];
    else
        target_onehot(i, :) = [0, 0, 1];
    end
end

% Set learning rate
learning_rate = 0.1; 

% Initialize weights with bias(+1 for bias)
% 10~12
Whid = rand(size(input, 2) + 1, 12) * 2 - 1;
Wout = rand(12 + 1, 3) * 2 - 1;

% Initialize RMSE array
RMSE = zeros(100, 3);

% Training
for epoch = 1:100
    squared_errors = zeros(1, 3);
    for iter = 1:75
        % Forward pass
        SUMhid = [input(iter, :) 1] * Whid;
        Ahid = logsig(SUMhid);
        SUMout = [Ahid 1] * Wout;
        Aout = purelin(SUMout);

        % Backpropagation
        DELTAout = (target_onehot(iter, :) - Aout);
        DELTAhid = DELTAout.*dpurelin(Aout)*Wout';
        
        % Update weights
        Wout = Wout + learning_rate * [Ahid 1]' * DELTAout;
        Whid(:,1) = Whid(:,1) + learning_rate * [input(iter, :) 1]' * DELTAhid(1) * dlogsig(SUMhid(1), Ahid(1));
        Whid(:,2) = Whid(:,2) + learning_rate * [input(iter, :) 1]' * DELTAhid(2) * dlogsig(SUMhid(2), Ahid(2));
        Whid(:,3) = Whid(:,3) + learning_rate * [input(iter, :) 1]' * DELTAhid(3) * dlogsig(SUMhid(3), Ahid(3));
        
        squared_errors = squared_errors + sum((target_onehot(iter, :) - Aout).^2);
    end

    RMSE(epoch, :) = sqrt(squared_errors / 75);
    % Convert RMSE to table
    RMSE_table_epoch = array2table(RMSE(epoch, :), 'VariableNames', {'RMSE_1', 'RMSE_2', 'RMSE_3'});
    RMSE_table(epoch, :) = RMSE_table_epoch;

end

% Calculate mean RMSE for each epoch
mean_RMSE = mean(RMSE, 3);

% Plot avg RMSE vs Epoch
plot(1:100, mean_RMSE);
xlabel('Epoch');
ylabel('Average RMSE');
title('Average RMSE vs Epoch');

% Testing
correct_count = 0;
Aout_table = table();

for iter = 76:length(input)
    % Forward pass
    Ahid_test = logsig([input(iter, :) 1] * Whid);
    Aout_test = purelin([Ahid_test 1] * Wout);
    Aout_test_epoch = array2table(Aout_test, 'VariableNames', {'Output_1', 'Output_2', 'Output_3'});
    Aout_table = [Aout_table; Aout_test_epoch];
    writetable(Aout_table, 'Aout_test.csv');

    % Check correctness
    [~, predicted_class] = max(Aout_test); 
    [~, actual_class] = max(target_onehot(iter, :));
    if predicted_class == actual_class
        correct_count = correct_count + 1;
    end
end

test_accuracy = correct_count / (150 - 75);
fprintf('Test accuracy: %.2f%%\n', test_accuracy * 100);
%% 
