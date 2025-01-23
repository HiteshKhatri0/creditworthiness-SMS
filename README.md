# creditworthiness-SMS
%% SMS Creditworthiness Algorithm in Octave

% Load and prepare the dataset
% Each SMS is a text string with labels: 1 (credit) or 0 (debit)
data = load('sms_data.txt'); % Assume 'sms_data.txt' contains labeled data
sms_texts = data(:, 1:end-1); % SMS messages
labels = data(:, end);        % Labels (1 = credit, 0 = debit)

% Preprocess SMS data
function [features] = preprocess_sms(sms_texts)
    vocab_list = load('vocab.txt'); % Vocabulary list (a file with common SMS words)
    num_sms = length(sms_texts);
    features = zeros(num_sms, length(vocab_list));

    for i = 1:num_sms
        words = strsplit(lower(sms_texts{i}), ' '); % Split SMS into words
        for j = 1:length(words)
            idx = find(strcmp(vocab_list, words{j})); % Check if the word is in vocab_list
            if ~isempty(idx)
                features(i, idx) = 1; % Mark the word in the feature vector
            end
        end
    end
end

% Convert SMS data to feature vectors
X = preprocess_sms(sms_texts);

% Add a column of ones to X for the bias term
X = [ones(size(X, 1), 1) X];

% Initialize parameters for logistic regression
[m, n] = size(X);
initial_theta = zeros(n, 1);

% Define the sigmoid function
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

% Define the cost function and gradient for logistic regression
function [J, grad] = costFunction(theta, X, y)
    m = length(y);
    h = sigmoid(X * theta);
    J = (1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h));
    grad = (1 / m) * (X' * (h - y));
end

% Train the logistic regression model using fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, labels)), initial_theta, options);

% Predict the category of new SMS messages
function pred = predict(theta, X)
    pred = sigmoid(X * theta) >= 0.5;
end

% Test the model with new SMS data
test_sms = {'Your account is credited with INR 5000'; 'Your debit card was used for INR 1200'};
test_features = preprocess_sms(test_sms);
test_features = [ones(size(test_features, 1), 1) test_features];
predictions = predict(theta, test_features);

% Display results
for i = 1:length(test_sms)
    fprintf('SMS: %s\n', test_sms{i});
    if predictions(i) == 1
        fprintf('Category: Credit\n\n');
    else
        fprintf('Category: Debit\n\n');
    end
end
