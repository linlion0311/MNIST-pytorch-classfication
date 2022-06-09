import matplotlib.pyplot as plt
model1="CNN"
model2="FC"
#---------------------------逐行讀取
model1 = open('{}.txt'.format(model1),"r")
model1_max_acc = model1.readlines()
model1_max_acc = [float(i) for i in model1_max_acc ]
print(model1_max_acc)
#-----------------------------
#---------------------------逐行讀取
model2 = open('{}.txt'.format(model2),"r")
model2_max_acc = model2.readlines()
model2_max_acc = [float(i) for i in model2_max_acc ]
print(model2_max_acc)
#-----------------------------
a=[]
for i in range(len(model2_max_acc)):
    a.append(int(i))
iter=a

# plt.plot(iter,model2_max_acc,color='r', label='max_acc')
# plt.legend()
# plt.show()


plt.title("MNIST Classfication YCLin") # title
plt.ylabel("Val_acc") # y label
plt.xlabel("iteration") # x label
plt.plot(iter, model1_max_acc,color='r', label='CNN_acc')
plt.plot(iter,model2_max_acc,color='b', label='FC_acc')
plt.legend()
# plt.yticks(np.arange(98,100,step=0.2))
plt.show()




s=[1000,1500,1600]

# b=[200,400,458]

    # f = open('{}.txt'.format(model))
    # text = f.read()
    # text=int(text)
    # a=[0,1,2,3]
    # print(text)
    # f.close()

# s.append(2000)
# s.append(2000)
# b.append(600)
# b.append(600)
# plt.plot(s,b)
# print(s)
# print(b)
# plt.plot(s, b,color='r', label='max_acc')
# plt.legend()
# plt.show()

# path = 'output.txt'
# f = open(path, 'w')
# for a in s:
#     d=float(a)
#     f.write(f"{d}\n")
#     print(a)
# f.close()
# b=[]
# f = open('output.txt')
# for line in f.readlines():
#     print(line)
#     b.append(int(line))
# print(b)
# f.close
