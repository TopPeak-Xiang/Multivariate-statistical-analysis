#��������
import numpy as np                           #�������������
np.set_printoptions(precision=4)             #����numpy�������
import pandas as pd                          #�������ݷ�����
pd.set_option('display.precision',4)         #����pandas�������
import matplotlib.pyplot as plt              #���ػ�����ͼ��
plt.rcParams['font.sans-serif']=['SimHei'];  #������������Ϊ����
plt.rcParams['axes.unicode_minus']=False;    #������ʾͼ��������

# 2.1.3 #�Զ����ֵ���㺯��
def xbar(x):  
    n=len(x)
    xm=sum(x)/n
    return(xm)
    
# 2.5.1 #����Ƶ������ֱ��ͼ����   
def freq(X,bins=10): 
    H=plt.hist(X,bins);
    a=H[1][:-1]; b=H[1][1:];
    f=H[0]; p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp])
    Freq.index=['[����','����)','Ƶ��','Ƶ��(%)','�ۼ�Ƶ��(%)']
    return(round(Freq.T,2))
    
# 5.3.1 #��񻯼��㺯��
def bz(x): 
    z=(x-x.min())/(x.max()-x.min())*60+40
    return(z)
    
# 5.3.2 #�жϾ���A��AHPȨ�ؼ���   
def AHP(A): 
    print('�жϾ���:\n',A)
    m=np.shape(A)[0];
    D=np.linalg.eig(A);    #����ֵ
    E=np.real(D[0][0]);    #�������� 
    ai=np.real(D[1][:,0]); #�������ֵ   
    W=ai/sum(ai)           #Ȩ�ع�һ��
    if(m>2):
        print('L_max=',E.round(4))
        CI=(E-m)/(m-1)   #����һ���Ա���
        RI=[0,0,0.52,0.89,1.12,1.25,1.35,1.42,1.46,1.49,1.52,1.54,1.56,1.58,1.59]
        CR=CI/RI[m-1]
        print('һ����ָ�� CI:',CI)
        print('һ���Ա��� CR:',CR)    
        if CR<0.1: print('CR<=0.1��һ���Կ��Խ���!')
        else: print('CR>0.1��һ���Բ��ɽ���!')
    print('Ȩ������:')
    return(W)

# 6.3.1 #���ɷ����ۺ���
def PCscores(X,m=2): 
    from sklearn.decomposition import PCA
    Z=(X-X.mean())/X.std()  #���ݱ�׼��
    p=Z.shape[1]
    pca = PCA(n_components=p).fit(Z)
    Vi=pca.explained_variance_;Vi
    Wi=pca.explained_variance_ratio_;Wi
    Vars=pd.DataFrame({'Variances':Vi});Vars  #,index=X.columns
    Vars.index=['Comp%d' %(i+1) for i in range(p)]
    Vars['Explained']=Wi*100;Vars
    Vars['Cumulative']=np.cumsum(Wi)*100;
    print("\n�����:\n",round(Vars,4))
    Compi=['Comp%d' %(i+1) for i in range(m)]
    loadings=pd.DataFrame(pca.components_[:m].T,columns=Compi,index=X.columns);
    print("\n���ɷָ���:\n",round(loadings,4))
    scores=pd.DataFrame(pca.fit_transform(Z)).iloc[:,:m];
    scores.index=X.index; scores.columns=Compi;scores
    scores['Comp']=scores.dot(Wi[:m]);scores
    scores['Rank']=scores.Comp.rank(ascending=False).astype(int);
    return scores   #print('\n�ۺϵ÷�������:\n',round(scores,4))

# 6.3.2 #�Զ���÷�ͼ���ƺ���
def Scoreplot(Scores): 
    plt.plot(Scores.iloc[:,0],Scores.iloc[:,1],'*'); 
    plt.xlabel(Scores.columns[0]);plt.ylabel(Scores.columns[1])
    plt.axhline(y=0,ls=':');plt.axvline(x=0,ls=':')
    for i in range(len(Scores)):
        plt.text(Scores.iloc[i,0],Scores.iloc[i,1],Scores.index[i])
        
# 7.2.1 #������������ 
def Factors(fa):  
    return ['F'+str(i) for i in range(1,fa.n_factors+1)] 

# 7.4.2 #˫��������Ϣ�ص�ͼ
def Biplot(Load,Score): 
    plt.plot(Scores.iloc[:,0],Scores.iloc[:,1],'*'); 
    plt.xlabel(Scores.columns[0]);plt.ylabel(Scores.columns[1])
    plt.axhline(y=0,ls=':');plt.axvline(x=0,ls=':')
    for i in range(len(Scores)):
        plt.text(Scores.iloc[i,0],Scores.iloc[i,1],Scores.index[i])

# 7.4.3 #�����ۺ����ӵ÷�������
def FArank(Vars,Scores): 
    Vi=Vars.values[0]
    Wi=Vi/sum(Vi);Wi
    Fi=Scores.dot(Wi)
    Ri=Fi.rank(ascending=False).astype(int);
    return(pd.DataFrame({'���ӵ÷�':Fi,'��������':Ri}))

# 7.5.2 #���ӷ����ۺ����ۺ���
def FAscores(X,m=2,rot='varimax'): 
    import factor_analyzer as fa
    kmo=fa.calculate_kmo(X) 
    chisq=fa.calculate_bartlett_sphericity(X) #����bartlett����
    print('KMO����: KMOֵ=%6.4f����ֵ=%8.4f, pֵ=%5.4f'% (kmo[1],chisq[0],chisq[1]))
    from factor_analyzer import FactorAnalyzer as FA
    Fp=FA(n_factors=m,method='principal',rotation=rot).fit(X.values)
    vars=Fp.get_factor_variance()
    Factor=['F%d' %(i+1) for i in range(m)]
    Vars=pd.DataFrame(vars,['����','������','�ۼƹ�����'],Factor)
    print("\n�����:\n",Vars)
    Load=pd.DataFrame(Fp.loadings_,X.columns,Factor) 
    Load['��ͬ��']=1-Fp.get_uniquenesses()
    print("\n�����غ�:\n",Load)
    Scores=pd.DataFrame(Fp.transform(X.values),X.index,Factor)    
    print("\n���ӵ÷�:\n",Scores)
    Vi=vars[0]
    Wi=Vi/sum(Vi);Wi
    Fi=Scores.dot(Wi)
    Ri=Fi.rank(ascending=False).astype(int);
    print("\n�ۺ�����:\n")
    return pd.DataFrame({'�ۺϵ÷�':Fi,'�ۺ�����':Ri})

# 9.2.1 #�����˻��ͺ���
def lxy(x,y):  
    return sum(x*y)-sum(x)*sum(y)/len(x)  

# 9.3.1 #���ϵ���������
import scipy.stats as st    #����ͳ�ư�
def mcor_test(X):     #���ϵ���������
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
    print("\n������Ϊ���ϵ����������Ϊ����pֵ * p<0.05 ** p<0.05 *** p<0.001")
     
# 10.3.4 #������ؼ��麯��
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
        Q[k]=(n-k-1/2*(p+q+3))*Q[k] #����Ŀ���ֵ
        P[k]=1-stats.chi2.cdf(Q[k],(p-k)*(q-k)) #Pֵ
    CR=DF({'CR':r,'Q':Q,'P':P})
    return CR

# 10.4.1 #������ط�������
def cancor(X,Y,pq=None,plot=False): #pqָ�����ͱ�������
    import numpy as np
    n,p=np.shape(X); n,q=np.shape(Y)
    if pq==None: pq=min(p,q)
    cca=CCA(n_components=pq).fit(X,Y); 
    u_scores,v_scores=cca.transform(X,Y) 
    r=DF(u_scores).corrwith(DF(v_scores));  
    CR=CR_test(n,p,q,r)           
    print('�������ϵ�����飺\n',CR)   
    print('\n������ر���ϵ����\n')
    u_coef=DF(cca.x_rotations_.T,['u%d'%(i+1) for i in range(pq)],X.columns)
    v_coef=DF(cca.y_rotations_.T,['v%d'%(i+1) for i in range(pq)],Y.columns)    
    if plot: #��ʾ��һ�Ե��ͱ����Ĺ�ϵͼ
        import matplotlib.pyplot as plt    
        plt.plot(u_scores[:,0],v_scores[:,0],'o')
    return u_coef,v_coef
    
# 12.2.2 #�����ʼ��㺯��
def Rate(tab): 
    rate=sum(np.diag(tab)[:-1]/np.diag(tab)[-1:])*100
    print('������: %.2f'%rate)     
    Rate(tab1)

