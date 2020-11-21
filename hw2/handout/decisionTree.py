import scipy as sp
import os
import numpy as np
import sys

class Node:
    def __init__(self, val):
        self.val = val
        self.index = None
        self.leftDict = dict()
        self.rightDict = dict()
        self.leftNode = None
        self.rightNode = None
        self.leftLabel = None
        self.rightLabel = None

def openFile(inputFile):
    with open(inputFile, 'rt') as raw_data:
        read_tsv = np.loadtxt(raw_data, dtype = np.str, delimiter = '\t')
        head_train = read_tsv[0]
        data_train = read_tsv[1:]
    return head_train, data_train

def findChoice(data, index):
    data = np.array(data)
    col = data[:,index].tolist()
    item1 = col[0]
    for i in range(len(col)):
        if col[i] != item1:
            item2 = col[i]
            return item1, item2
    item2 = None
    return item1, item2
    
    

def giniImpurityCal(data, index):
    sum_rows = data.shape[0]
    size = data.shape[1]
    if (sum_rows == 0):
        return 0
    else:
        item1, item2 = findChoice(data, index)
        group_one = data[np.where(data[:, index] == item1)]
        group_two = data[np.where(data[:, index] == item2)]
        prob_a = group_one.shape[0] / sum_rows
        prob_b = group_two.shape[0] / sum_rows
        gini_impurity = prob_a*(1 - prob_a) + prob_b*(1 - prob_b)
        return gini_impurity
    


def bestNodeFinder(head, data):
    if data.shape[0] == 0:
        return None
    else:
        sum_rows = data.shape[0]
        size = data.shape[1]
        best_gain = -1
        best_index = -1
        gini_impurity = giniImpurityCal(data, -1)
        item1, item2 = findChoice(data, 0)
        type1, type2 = findChoice(data, size-1)
        for index in range(0, size-1):
            group_one = data[np.where(data[:, index] == item1)]
            group_two = data[np.where(data[:, index] == item2)]
            prob_one = group_one.shape[0] / sum_rows
            prob_two = group_two.shape[0] / sum_rows
            if (group_one.shape[0] != 0 and group_two.shape[0] != 0):
                group_one_imp = giniImpurityCal(group_one, -1)
                group_two_imp = giniImpurityCal(group_two, -1)
                gini_gain = gini_impurity - (prob_one*group_one_imp + prob_two*group_two_imp)
            elif (group_one.shape[0] == 0 and group_two.shape[0] != 0):
                group_one_imp = 0
                group_two_imp = giniImpurityCal(group_two, -1)
                gini_gain = gini_impurity - (prob_two*group_two_imp)
            elif (group_one.shape[0] != 0 and group_two.shape[0] == 0):
                group_two_imp = 0
                group_one_imp = giniImpurityCal(group_one, -1)
                gini_gain = gini_impurity - (prob_one*group_one_imp)
            else:
                continue

            if gini_gain > best_gain:
                best_gain = gini_gain
                best_index = index

        best_title = head[best_index]
        best_node = Node(best_title)
        best_node.index = best_index
        group1 = data[np.where((data[:, best_index] == item1) & (data[:, size-1] == type1))]
        group2 = data[np.where((data[:, best_index] == item1) & (data[:, size-1] == type2))]
        group3 = data[np.where((data[:, best_index] == item2) & (data[:, size-1] == type1))]
        group4 = data[np.where((data[:, best_index] == item2) & (data[:, size-1] == type2))]
        best_node.leftDict[type1] = group1.shape[0]
        best_node.leftDict[type2] = group2.shape[0]
        best_node.leftLabel = item1
        best_node.rightDict[type1] = group3.shape[0]
        best_node.rightDict[type2] = group4.shape[0]
        best_node.rightLabel = item2
        return best_node

def decisionTreeTrain(head, data, depth, max_depth):
    if max_depth == 0:
        sum_rows = data.shape[0]
        size = data.shape[1]
        type1, type2 = findChoice(data, size-1)
        group_one = data[np.where(data[:, size-1] == type1)]
        group_two = data[np.where(data[:, size-1] == type2)]
        if group_one.shape[0] > group_two.shape[0]:
            bestNode = Node(type1)
            return bestNode
        elif group_one.shape[0] < group_two.shape[0]:
            bestNode = Node(type2)
            return bestNode
        else:
            if type1 > type2:
                bestNode = Node(type1)
            else:
                bestNode = Node(type2)
            return bestNode
    if depth == max_depth - 1:
        maxValue = 0
        bestNode = bestNodeFinder(head, data)
        if bestNode == None:
            return None
        else:
            leftName = list(bestNode.leftDict)
            leftValue = list(bestNode.leftDict.values())
            rightName = list(bestNode.rightDict)
            rightValue = list(bestNode.rightDict.values())
            if (rightValue[0] == rightValue[1]):
                if None in rightName:
                    rightName.remove(None)
                selected = max(rightName)
                bestNode.rightNode = Node(selected)
            else:
                maxParty_right = max(bestNode.rightDict, key = bestNode.rightDict.get)
                bestNode.rightNode = Node(maxParty_right)
            if (leftValue[0] == leftValue[1]):
                if None in leftName:
                    leftName.remove(None)
                selected = max(leftName)
                bestNode.leftNode = Node(selected)
            else:
                maxParty_left = max(bestNode.leftDict, key = bestNode.leftDict.get)
                bestNode.leftNode = Node(maxParty_left)
            return bestNode

    else:
        bestNode = bestNodeFinder(head, data)
        if bestNode == None:
            return None
        else:
            leftData = data[np.where((data[:, bestNode.index] == bestNode.leftLabel))]
            rightData = data[np.where((data[:, bestNode.index] == bestNode.rightLabel))]
            head = np.delete(head, bestNode.index, axis = 0)
            leftData = np.delete(leftData, bestNode.index, axis = 1)
            rightData = np.delete(rightData, bestNode.index, axis = 1)
            if (leftData.shape[0] != 0):
                bestNode.leftNode = decisionTreeTrain(head, leftData, depth+1, max_depth)
            else:
                bestNode.leftNode = Node(list(rightData[:,-1])[0])
                bestNode.rightNode = bestNode.leftNode
            if (rightData.shape[0] != 0):
                bestNode.rightNode = decisionTreeTrain(head, rightData, depth+1, max_depth)
            else:
                bestNode.leftNode = Node(list(leftData[:,-1])[0])
                bestNode.rightNode = bestNode.leftNode
            return bestNode

def decisionTreeTest(head, data, node, max_depth):
    if (max_depth == 0):
        size = len(data)
        output = [node.val]*size
        return output
    else:
        output = []
        for i in range(len(data)):
            if (i == 23):
                row = data[i]
                print (i, row)
                result = decisionTreeTestWrappedPrint(data, row, head, node)
                output.append(result)
                print (len(output))
            else:
                row = data[i]
                result = decisionTreeTestWrapped(data, row, head, node)
                output.append(result)
        return output
def decisionTreeTestWrappedPrint(data, row, headList, node):
    if node is None:
        return None
    if node.val not in headList:
        return node.val
    else:
        loc = headList.index(node.val)
        if row[loc] == node.leftLabel:
            result = decisionTreeTestWrapped(data, row, headList, node.leftNode)
        elif row[loc] == node.rightLabel:
            result = decisionTreeTestWrapped(data, row, headList, node.rightNode)
        else:
            type1, type2 = findChoice(data, -1)
            if node.leftLabel == None:
                result = type1 if (node.rightLabel == type2) else type2
            else:
                result = type2 if (node.leftLabel == type1) else type1
        return result
        
def decisionTreeTestWrapped(data, row, headList, node):
    if node is None:
        return None
    if node.val not in headList:
        return node.val
    else:
        loc = headList.index(node.val)
        if row[loc] == node.leftLabel:
            result = decisionTreeTestWrapped(data, row, headList, node.leftNode)
        elif row[loc] == node.rightLabel:
            result = decisionTreeTestWrapped(data, row, headList, node.rightNode)
        else:
            type1, type2 = findChoice(data, -1)
            if node.leftLabel == None:
                result = type1 if (node.rightLabel == type2) else type2
            else:
                result = type2 if (node.leftLabel == type1) else type1
        return result

def errorRate(output, data):
    data = np.array(data)
    diff = 0
    colsize = data.shape[1]
    result = data[:,colsize-1]
    result = result.tolist()
    for i in range(len(output)):
        if output[i] != result[i]:
            diff += 1
    return (diff/len(output))

def outputFile(inputFile, output, train_out, test_out, metrics_out):
    if 'train' in inputFile:
        np.savetxt(train_out, output, fmt = '%s', newline = '\n')
        errorText = open(metrics_out, 'a')
        errorRate_train = errorRate(output, data)
        errorText.write(f'error(train): {errorRate_train}' + '\n')
        errorText.close()
    elif 'test' in inputFile:
        np.savetxt(test_out, output, fmt = '%s', newline = '\n')
        errorText = open(metrics_out, 'a')
        errorRate_test = errorRate(output, data)
        errorText.write(f'error(test): {errorRate_test}' + '\n')
        errorText.close()

def printPreorder(data, root):
    data = np.array(data)
    type1, type2 = findChoice(data, -1)
    groupA = data[np.where(data[:, -1] == type1)].shape[0]
    groupB = data[np.where(data[:, -1] == type2)].shape[0]
    print (f'[{groupA} {type1} /{groupB} {type2}]')
    recursivePrint(root, type1, type2)
def recursivePrint(root, type1, type2, depth = '| '):
    if root:
        if root.leftDict != {}:
            print (depth+f'{root.val} = {root.leftLabel}: [{list(root.leftDict.values())[0]} {list(root.leftDict)[0]} /{list(root.leftDict.values())[1]} {list(root.leftDict)[1]}]')
            recursivePrint(root.leftNode, type1, type2, depth+'| ')
        if root.rightDict != {}:
            print (depth+f'{root.val} = {root.rightLabel}: [{list(root.rightDict.values())[0]} {list(root.rightDict)[0]} /{list(root.rightDict.values())[1]} {list(root.rightDict)[1]}]')
            recursivePrint(root.rightNode, type1, type2, depth+'| ')


def attributeDepthCheck(depth, data):
    colnum = data.shape[1] - 1
    if (colnum < depth):
        return colnum
    else:
        return depth




        
if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_index = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    max_index = int(max_index)
    f = open(metrics_out, 'a')
    f.seek(0)
    f.truncate()
    f.close()
    head, data = openFile(train_input)
    max_index = attributeDepthCheck(max_index, data)
    node = decisionTreeTrain(head, data, 0, max_index)
    printPreorder(data, node)
    head = head.tolist()
    data = data.tolist()
    output = decisionTreeTest(head, data, node, max_index)
    outputFile1 = outputFile(train_input, output, train_out, test_out, metrics_out)
    head, data = openFile(test_input)
    head = head.tolist()
    data = data.tolist()
    output = decisionTreeTest(head, data, node, max_index)
    outputFile1 = outputFile(test_input, output, train_out, test_out, metrics_out)