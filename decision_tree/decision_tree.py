import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.tree = None
    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels

        Returns:
            A decision tree, represented as a nested dictionary.
        """
         
        if len(np.unique(y)) == 1:  # If all labels are the same, return a leaf node
            # print("All labels are the same, returning a leaf node ", y.iloc[0])
            return y.iloc[0] # Return the first label in y (they're all the same)

        best_feature, best_split = self.select_best_split(X, y)
        tree = {best_feature: {}} # Create a new tree node (dictionary) with the best feature as the key
        for value in best_split: # For each unique value of the best feature (i.e. 'Weak', 'Strong' for 'Wind')
            subset_X = X[X[best_feature] == value] # Find the elements of X that correspond to the current feature value (e.g. 'Weak')
            subset_y = y[X[best_feature] == value] # Find the elements of y that correspond to the current feature value (e.g. 'Weak')
            # Recursively call this function, passing in the current subset of X and y, and add the resulting subtree to the current tree:
            tree[best_feature][value] = self.fit(subset_X.drop(columns=[best_feature]), subset_y) 

        return tree

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        predictions = []
        
        for _, row in X.iterrows(): # Iterate over rows in the input DataFrame
            current_node = self.tree 
            
            while isinstance(current_node, dict): # While the current node is not a leaf node (because it is a dictionary, not a string)
                feature = list(current_node.keys())[0] # Get the feature to split on (the first key in the dictionary)
                
                if row[feature] in current_node[feature]: # If the feature value matches a branch
                    current_node = current_node[feature][row[feature]] # Move to the next node
                else:
                    break # If there's no matching branch, exit the loop
            
            predictions.append(current_node)
            
        return predictions
    

    def select_best_split(self, X, y):
        '''
        Selects the best feature to split on based on information gain (ID3 algorithm)

        Args:
            X (pd.DataFrame): the input data
            y (pd.Series): a vector of discrete ground-truth labels (i.e. 'Yes' or 'No')

        Returns:
            best_feature (str): the feature to split on in order to maximize information gain
            best_split (list): the unique values of the best feature (i.e. 'Weak', 'Strong' for 'Wind')
        '''
        best_feature = None
        best_info_gain = -1

        for feature in X.columns: # For each feature in the input X DataFrame
            info_gain = self.get_information_gain(X, y, feature) # Calculate the information gain of the feature
            if info_gain > best_info_gain: # If the information gain is greater than the current best information gain
                best_info_gain = info_gain # Update the best information gain
                best_feature = feature # Update the best feature to split on

        if best_feature is not None: # If a best feature was found
            best_split = np.unique(X[best_feature]) # Get the unique values of the best feature (i.e. 'Weak', 'Strong' for 'Wind')
        else:
            best_split = [] # If no best feature was found, return an empty list

        return best_feature, best_split

    def get_information_gain(self, X, y, feature):
        '''
        Calculates the information gain of a feature by comparing the entropy of the parent node to the entropy of the children nodes

        Args:
            X (pd.DataFrame): the input data
            y (pd.Series): a vector of discrete ground-truth labels (i.e. 'Yes' or 'No')
            feature (str): the feature to split on (i.e. could be 'Outlook', 'Humidity', 'Wind', or 'Temperature' in the first dataset)

        Returns:
            information_gain (float): the information gain of the feature (i.e. how much the entropy decreases by splitting on the feature)
        '''
        counts = y.value_counts() # Get the counts of each label in the parent node (i.e. how many occurrences of 'Yes' and 'No')
        entropy_parent = entropy(counts) # Calculate the entropy of the parent node
        entropy_children = 0 

        for value in np.unique(X[feature]): # For each unique value of the feature (i.e. 'Weak', 'Strong' for 'Wind')
            subset_y = y[X[feature] == value] # Find the elements of y that correspond to the current feature value (e.g. 'Weak')
            counts_child = subset_y.value_counts() # Get the counts of each label in the child node (i.e. how many occurrences of 'Yes' and 'No')
            entropy_children += (len(subset_y) / len(y)) * entropy(counts_child) # Calculate the entropy of the chosen feature to split on

        return entropy_parent - entropy_children 

    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # Because self.tree is a nested dictionary, we use a recursive function to extract the rules so it matches the expected output format
        def extract_rules(tree, rule=None):
            if rule is None: # If no rule is passed in, initialize an empty list
                rule = []
            if isinstance(tree, dict): # Check if the current node is a dictionary (i.e. not a leaf node, meaning there are more tree levels to traverse)
                feature = list(tree.keys())[0] # Start with the first feature in the dictionary

                for value, sub_tree in tree[feature].items(): # For each "level" under the current feature
                    new_rule = rule + [(feature, value)] # Making sure that we add to the existing rule, to avoid splitting the same rule into multiple rules
                    yield from extract_rules(sub_tree, new_rule) # Recursively call this function, passing in the current subtree and the new rule
            else:
                # print("Encountered a leaf node given rule: ", rule)
                yield (rule, tree) # Return the rule and the tree as it was passed in (happens when the current node is a leaf node)

        return list(extract_rules(self.tree)) # Return the list of rules
    
# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    # Convert to pandas vector (expected argument type)
    y_true = pd.Series(y_true).reset_index(drop=True)  # Reset the indices to match (comparison failed for dataset 2 without this)
    y_pred = pd.Series(y_pred).reset_index(drop=True)  
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


if __name__ == "__main__":
    # Notebook code for dataset 1
    print("\n##################### DATASET 1 #####################")

    data_1 = pd.read_csv('decision_tree/data_1.csv')
    X = data_1.drop(columns=['Play Tennis'])
    y = data_1['Play Tennis']

    model_1 = DecisionTree()
    model_1.tree = model_1.fit(X, y)

    y_pred = model_1.predict(X)
    print(f'Accuracy: {accuracy(y, y_pred) * 100:.1f}%')

    for rules, label in model_1.get_rules():
        conjunction = ' ∩ '.join(f'{attr}={value}' for attr, value in rules)
        print(f'{"✅" if label == "Yes" else "❌"} {conjunction} => {label}')

    # Notebook code for dataset 2
    print("\n##################### DATASET 2 #####################")
    data_2 = pd.read_csv('decision_tree/data_2.csv')

    data_2_train = data_2.query('Split == "train"')
    data_2_valid = data_2.query('Split == "valid"')
    data_2_test = data_2.query('Split == "test"')
    X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
    X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
    X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome
    data_2.Split.value_counts()

    # Fit model (TO TRAIN SET ONLY)
    model_2 = DecisionTree()  # <-- Feel free to add hyperparameters 
    # model_2 = model_2.fit(X_train, y_train)
    
    # NB
    model_2.tree = model_2.fit(X_train, y_train) 
    #print(y_valid)
    #print("\n", model_2.predict(X_valid), "\n")
    #print("Length of y_valid: ", len(y_valid), ", Length of predictions: ", len(model_2.predict(X_valid)))
    # NB

    print(f'Train: {accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
    print(f'Valid: {accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')

    
