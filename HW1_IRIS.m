%{
CE8009 -圖形識別實務與應用
土木4B 109302545 莊明儒
%}

function HW1_IRIS_109302545

% Load input and target data
load IRIS_IN.csv;
load IRIS_OUT.csv;
input = IRIS_IN;
target = IRIS_OUT;

% Set learning rate
learning_rate = 0.01;

% Initialize weights with bias(+1 for bias)
Whid = rand(size(input, 2) + 1, 12) * 2 - 1;
Wout = rand(12 + 1, 1) * 2 - 1;

% Initialize RMSE array
RMSE = zeros(100, 1);

% Training
for epoch = 1:100
    squared_errors = 0;
    for iter = 1:75
        % Forward pass
        SUMhid = [input(iter, :) 1] * Whid;
        Ahid = logsig(SUMhid);
        SUMout = [Ahid 1] * Wout;
        Aout = purelin(SUMout);

        % Backpropagation
        % .* -> 對於神經網絡中的每個隱藏單元，都將對應的相應值相乘
        DELTAout = target(iter) - Aout;
        DELTAhid = DELTAout * dpurelin(Aout) * Wout(1:end-1)' .* dpurelin(Ahid);

        % Update weights
        Wout = Wout + learning_rate * [Ahid 1]' * DELTAout;
        Whid(:,1) = Whid(:,1) + learning_rate * [input(iter, :) 1]' * DELTAhid(1) * dlogsig(SUMhid(1), Ahid(1));
        Whid(:,2) = Whid(:,2) + learning_rate * [input(iter, :) 1]' * DELTAhid(2) * dlogsig(SUMhid(2), Ahid(2));

        squared_errors = squared_errors + (target(iter) - Aout)^2;
    end
    RMSE(epoch) = sqrt(squared_errors / 75);
    fprintf('Epoch %d: RMSE = %.3f\n', epoch, RMSE(epoch));
end

% Plot RMSE vs Epoch
plot(1:100, RMSE);
xlabel('Epoch');
ylabel('RMSE');
title('RMSE vs Epoch');

% Testing
correct_count = 0;
for iter = 76:length(input)
    % Forward pass
    Ahid_test = logsig([input(iter, :) 1] * Whid);
    Aout_test = purelin([Ahid_test 1] * Wout);

    % Print output
    fprintf('Test Data point %d: Output = %.3f\n', iter, Aout_test);
    
    % Check correctness
    if abs(Aout_test - target(iter)) <= 0.5
        correct_count = correct_count + 1;
    end
end

test_accuracy = correct_count / (150 - 75);
fprintf('Test accuracy: %.2f%%\n', test_accuracy * 100);

end
