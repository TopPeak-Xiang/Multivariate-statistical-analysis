#基本设置
import numpy as np                           #加载数组运算包
np.set_printoptions(precision=4)             #设置numpy输出精度
import pandas as pd                          #加载数据分析包
pd.set_option('display.precision',4)         #设置pandas输出精度
import matplotlib.pyplot as plt              #加载基本绘图包
plt.rcParams['font.sans-serif']=['SimHei'];  #设置中文字体为黑体
plt.rcParams['axes.unicode_minus']=False;    #正常显示图中正负号

# 2.1.3 #自定义均值计算函数
def xbar(x):  
    n=len(x)
    xm=sum(x)/n
    return(xm)
    
# 2.5.1 #定量频数表与直方图函数   
def freq(X,bins=10): 
    H=plt.hist(X,bins);
    a=H[1][:-1]; b=H[1][1:];
    f=H[0]; p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp])
    Freq.index=['[下限','上限)','频数','频率(%)','累计频数(%)']
    return(round(Freq.T,2))
    
# 5.3.1 #规格化计算函数
def bz(x): 
    z=(x-x.min())/(x.max()-x.min())*60+40
    return(z)
    
# 5.3.2 #判断矩阵A的AHP权重计算   
def AHP(A): 
    print('判断矩阵:\n',A)
    m=np.shape(A)[0];
    D=np.linalg.eig(A);    #特征值
    E=np.real(D[0][0]);    #特征向量 
    ai=np.real(D[1][:,0]); #最大特征值   
    W=ai/sum(ai)           #权重归一化
    if(m>2):
        print('L_max=',E.round(4))
        CI=(E-m)/(m-1)   #计算一致性比例
        RI=[0,0,0.52,0.89,1.12,1.25,1.35,1.42,1.46,1.49,1.52,1.54,1.56,1.58,1.59]
        CR=CI/RI[m-1]
        print('一致性指标 CI:',CI)
        print('一致性比例 CR:',CR)    
        if CR<0.1: print('CR<=0.1，一致性可以接受!')
        else: print('CR>0.1，一致性不可接受!')
    print('权重向量:')
    return(W)

# 6.3.1 #主成分评价函数
def PCscores(X,m=2): 
    from sklearn.decomposition import PCA
    Z=(X-X.mean())/X.std()  #数据标准化
    p=Z.shape[1]
    pca = PCA(n_components=p).fit(Z)
    Vi=pca.explained_variance_;Vi
    Wi=pca.explained_variance_ratio_;Wi
    Vars=pd.DataFrame({'Variances':Vi});Vars  #,index=X.columns
    Vars.index=['Comp%d' %(i+1) for i in range(p)]
    Vars['Explained']=Wi*100;Vars
    Vars['Cumulative']=np.cumsum(Wi)*100;
    print("\n方差贡献:\n",round(Vars,4))
    Compi=['Comp%d' %(i+1) for i in range(m)]
    loadings=pd.DataFrame(pca.components_[:m].T,columns=Compi,index=X.columns);
    print("\n主成分负荷:\n",round(loadings,4))
    scores=pd.DataFrame(pca.fit_transform(Z)).iloc[:,:m];
    scores.index=X.index; scores.columns=Compi;scores
    scores['Comp']=scores.dot(Wi[:m]);scores
    scores['Rank']=scores.Comp.rank(ascending=False).astype(int);
    return scores   #print('\n综合得分与排名:\n',round(scores,4))

# 6.3.2 #自定义得分图绘制函数
def Scoreplot(Scores): 
    plt.plot(Scores.iloc[:,0],Scores.iloc[:,1],'*'); 
    plt.xlabel(Scores.columns[0]);plt.ylabel(Scores.columns[1])
    plt.axhline(y=0,ls=':');plt.axvline(x=0,ls=':')
    for i in range(len(Scores)):
        plt.text(Scores.iloc[i,0],Scores.iloc[i,1],Scores.index[i])
        
# 7.2.1 #定义因子名称 
def Factors(fa):  
    return ['F'+str(i) for i in range(1,fa.n_factors+1)] 

# 7.4.2 #双向因子信息重叠图
def Biplot(Load,Score): 
    plt.plot(Scores.iloc[:,0],Scores.iloc[:,1],'*'); 
    plt.xlabel(Scores.columns[0]);plt.ylabel(Scores.columns[1])
    plt.axhline(y=0,ls=':');plt.axvline(x=0,ls=':')
    for i in range(len(Scores)):
        plt.text(Scores.iloc[i,0],Scores.iloc[i,1],Scores.index[i])

# 7.4.3 #计算综合因子得分与排名
def FArank(Vars,Scores): 
    Vi=Vars.values[0]
    Wi=Vi/sum(Vi);Wi
    Fi=Scores.dot(Wi)
    Ri=Fi.rank(ascending=False).astype(int);
    return(pd.DataFrame({'因子得分':Fi,'因子排名':Ri}))

# 7.5.2 #因子分析综合评价函数
def FAscores(X,m=2,rot='varimax'): 
    import factor_analyzer as fa
    kmo=fa.calculate_kmo(X) 
    chisq=fa.calculate_bartlett_sphericity(X) #进行bartlett检验
    print('KMO检验: KMO值=%6.4f卡方值=%8.4f, p值=%5.4f'% (kmo[1],chisq[0],chisq[1]))
    from factor_analyzer import FactorAnalyzer as FA
    Fp=FA(n_factors=m,method='principal',rotation=rot).fit(X.values)
    vars=Fp.get_factor_variance()
    Factor=['F%d' %(i+1) for i in range(m)]
    Vars=pd.DataFrame(vars,['方差','贡献率','累计贡献率'],Factor)
    print("\n方差贡献:\n",Vars)
    Load=pd.DataFrame(Fp.loadings_,X.columns,Factor) 
    Load['共同度']=1-Fp.get_uniquenesses()
    print("\n因子载荷:\n",Load)
    Scores=pd.DataFrame(Fp.transform(X.values),X.index,Factor)    
    print("\n因子得分:\n",Scores)
    Vi=vars[0]
    Wi=Vi/sum(Vi);Wi
    Fi=Scores.dot(Wi)
    Ri=Fi.rank(ascending=False).astype(int);
    print("\n综合排名:\n")
    return pd.DataFrame({'综合得分':Fi,'综合排名':Ri})

# 9.2.1 #离均差乘积和函数
def lxy(x,y):  
    return sum(x*y)-sum(x)*sum(y)/len(x)  

# 9.3.1 #相关系数矩阵检验
import scipy.stats as st    #加载统计包
def mcor_test(X):     #相关系数矩阵检验
    p=X.shape[1];p
    sp=np.ones([p, p]).astype(str)
    for i in range(0,p):
        for j in range(i,p):        
            P=st.pearsonr(X.iloc[:,i],X.iloc[:,j])[1]        
            if P>0.05: sp[i,j]=' '
            if(P>0.01 or P<=0.05): sp[i,j]='*'
            if(P>0.001 or P<=0.01): sp[i,j]='**'
            if(P<=0.001): sp[i,j]='***'
            r=st.pearsonr(X.iloc[:,i],X.iloc[:,j])[0]
            sp[j,i]=round(r,4)
            if(i==j):sp[i,j]='------'    
    print(pd.DataFrame(sp,index=X.columns,columns=X.columns))
    print("\n下三角为相关系数，上三角为检验p值 * p<0.05 ** p<0.05 *** p<0.001")
     
# 10.3.4 #典型相关检验函数
def CR_test(n,p,q,r):  
    m=len(r); 
    import numpy as np
    Q=np.zeros(m); P=np.zeros(m)
    L=1  #lambda=1
    from math import log
    for k in range(m-1,-1,-1):  
        L=L*(1-r[k]**2)  
        Q[k]=-log(L)
    from scipy import stats                
    for k in range(0,m):
        Q[k]=(n-k-1/2*(p+q+3))*Q[k] #检验的卡方值
        P[k]=1-stats.chi2.cdf(Q[k],(p-k)*(q-k)) #P值
    CR=DF({'CR':r,'Q':Q,'P':P})
    return CR

# 10.4.1 #典型相关分析函数
def cancor(X,Y,pq=None,plot=False): #pq指定典型变量个数
    import numpy as np
    n,p=np.shape(X); n,q=np.shape(Y)
    if pq==None: pq=min(p,q)
    cca=CCA(n_components=pq).fit(X,Y); 
    u_scores,v_scores=cca.transform(X,Y) 
    r=DF(u_scores).corrwith(DF(v_scores));  
    CR=CR_test(n,p,q,r)           
    print('典型相关系数检验：\n',CR)   
    print('\n典型相关变量系数：\n')
    u_coef=DF(cca.x_rotations_.T,['u%d'%(i+1) for i in range(pq)],X.columns)
    v_coef=DF(cca.y_rotations_.T,['v%d'%(i+1) for i in range(pq)],Y.columns)    
    if plot: #显示第一对典型变量的关系图
        import matplotlib.pyplot as plt    
        plt.plot(u_scores[:,0],v_scores[:,0],'o')
    return u_coef,v_coef
    
# 12.2.2 #符合率计算函数
def Rate(tab): 
    rate=sum(np.diag(tab)[:-1]/np.diag(tab)[-1:])*100
    print('符合率: %.2f'%rate)     
    Rate(tab1)

