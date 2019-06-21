###########
#1.Recap
###for continous :y=wx+b
###for probabilty output :y=sigmoid(wx+b)
####sigmoid==(logistic)

#2.Binary Classification
###interpret network as f:x->p(y|x;[w,b])
###output属于[0,1]
###which is exactly what logistic function comes in!

#3.Goal vs Approach
##For Regression
####Goal :pred=y
####Approach: minimize dist(pred,y)
##For classification
####Goal:maximize benchmark ,eg.accuracy
####Approach1:minimize dist(p(y|x)(labels),p(y|x)(output of network))
####Approach1:minimize divergence(p(y|x), p(y|x))

##Q1:why not maximize accuracy
###概念：acc=E(pred,y)/len(y)
###issues 1:gradient=0,if accuracy unchanged but weights changed
###issues 2:gradient not continuous since the number of correct is not continuous

##Q2:why call logistic regression
###use sigmoid
###Controversial!
#####MSE => regression
#####Cross Entropy => classification

#4.Multi-class classification
##softmax
####可以使Epi=1
####使大的更大

































