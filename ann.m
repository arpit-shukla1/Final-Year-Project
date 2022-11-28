%% training and development of ann
hiddenLayerSize= 13;
net=fitnet(hiddenLayerSize);
net.divideParam.trainRatio=70/100;
net.divideParam.valRatio=15/100;
net.divideParam.testRatio=15/100;
net.trainParam.epochs=5000;
net.trainParam.goal=0.00;
%%net.trainParam.lr=0.01;
[net,tr]=train(net,A,Y);





%% performance 
Ytrain = net(A(:,tr.trainInd));
Ytraintrue= Y(tr.trainInd);
sqrt(mean((Ytrain-Ytraintrue).^2))
Yval = net(A(:,tr.valInd));
Yvaltrue= Y(tr.valInd);
sqrt(mean((Yval-Yvaltrue).^2))

%%  visualisation

plot(Ytraintrue,Ytrain,'x') ; hold on ;
plot(Yvaltrue,Yval,'o');
plot(0:100,0:100); hold off ;


%% optimise the number of nuerons in hidden layer 

for i = 1: 20 
    
        
        
    
    %defining the architecture 
    hiddenLayerSize=i;
    net=fitnet(hiddenLayerSize);
    net.divideParam.trainRatio=70/100;
    net.divideParam.valRatio=15/100;
    net.divideParam.testRatio=15/100;
    % training the architecture 
    [net,tr]=train(net,A,Y);  
    
    % error calculations 
    
    Ytrain = net(A(:,tr.trainInd));
    Yval = net(A(:,tr.valInd));
   
    Ytraintrue= Y(tr.trainInd);
    Yvaltrue= Y(tr.valInd);
    rmsetrain(i)=sqrt(mean((Ytrain-Ytraintrue).^2)); % error for training set 
    rmseval(i)=sqrt(mean((Yval-Yvaltrue).^2));    % error for val set 
    
    
    
    
    
end 
%%  selecting optimal values of neurons

plot(1:20 , rmsetrain); hold on;
plot(1:20, rmseval); hold off;
%% using ann 
test = net(A(:,tr.testInd));
a=sim(net,test');
target=Y(tr.testInd);
%% 
plot(target,a,'x') ; hold on ;
plot(0:100,0:100); hold off ;
    
