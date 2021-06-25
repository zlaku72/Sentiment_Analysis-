from string import punctuation, digits
import numpy as np
import random



# Part I


#pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):

     y = np.dot(theta, feature_vector) + theta_0
     loss = max(0.0, 1 - y * label)
     return loss

#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_full(feature_matrix, labels, theta, theta_0):

        loss = 0
        for i in range(len(feature_matrix)):
            loss += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
        return loss / len(labels)


#pragma: coderesponse end


#pragma: coderesponse template
def perceptron_single_step_update(
            feature_vector,
            label,
            current_theta,
            current_theta_0):
     if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 0:
        current_theta += label * feature_vector
        current_theta_0 += label
     return (current_theta, current_theta_0)

#pragma: coderesponse end


#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
        (nsamples, nfeatures) = feature_matrix.shape
        theta = np.zeros(nfeatures)
        theta_0 = 0.0

        for t in range(T):
          for i in get_order(feature_matrix.shape[0]):
             # Your code here
             theta, theta_0 = perceptron_single_step_update(
                 feature_matrix[i], labels[i], theta, theta_0)
        return (theta, theta_0)
pass
#pragma: coderesponse end


#pragma: coderesponse template
def average_perceptron(feature_matrix, labels, T):
     (nsamples, nfeatures) = feature_matrix.shape
     theta = np.zeros(nfeatures)
     theta_sum = np.zeros(nfeatures)
     theta_0 = 0.0
     theta_0_sum = 0.0
     for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
     return (theta_sum / (nsamples * T), theta_0_sum / (nsamples * T))

#pragma: coderesponse end


#pragma: coderesponse template
def pegasos_single_step_update(
            feature_vector,
            label,
            L,
            eta,
            current_theta,
            current_theta_0):
     mult = 1 - (eta * L)
     if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        return ((mult * current_theta) + (eta * label * feature_vector),
                (current_theta_0) + (eta * label))
     return (mult * current_theta, current_theta_0)

#pragma: coderesponse end


#pragma: coderesponse template
def pegasos(feature_matrix, labels, T, L):
        (nsamples, nfeatures) = feature_matrix.shape
        theta = np.zeros(nfeatures)
        theta_0 = 0
        count = 0
        for t in range(T):
            for i in get_order(nsamples):
                count += 1
                eta = 1.0 / np.sqrt(count)
                (theta, theta_0) = pegasos_single_step_update(
                    feature_matrix[i], labels[i], L, eta, theta, theta_0)
        return (theta, theta_0)

#pragma: coderesponse end

# Part II


#pragma: coderesponse template
def classify(feature_matrix, theta, theta_0):
    (nsamples, nfeatures) = feature_matrix.shape
    predictions = np.zeros(nsamples)
    for i in range(nsamples):
        feature_vector = feature_matrix[i]
        prediction = np.dot(theta, feature_vector) + theta_0
        if (prediction > 0):
            predictions[i] = 1
        else:
            predictions[i] = -1
    return predictions

#pragma: coderesponse end


#pragma: coderesponse template
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):

    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(val_predictions, val_labels)
    return (train_accuracy, validation_accuracy)

#pragma: coderesponse end


#pragma: coderesponse template
def extract_words(input_string):

    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
#pragma: coderesponse end


#pragma: coderesponse template
def bag_of_words(texts):
    stop_words = {}
    with open("stopwords.txt") as f_stop:
     for line in f_stop:
        s_line = line.rstrip()
        stop_words[s_line] = len(stop_words)

    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
             if word  in stop_words:continue
             if word not in dictionary:
                dictionary[word] = len(dictionary)

    return dictionary

#pragma: coderesponse end


#pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):



    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    return feature_matrix
#pragma: coderesponse end



#pragma: coderesponse template
def accuracy(preds, targets):

    return (preds == targets).mean()
#pragma: coderesponse end
