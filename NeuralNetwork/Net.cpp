//
//  Net.cpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 30/01/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//
#include "Net.hpp"


template <typename T>
void Net<T>::BeforNewEpoch(unsigned int iternum){
    for(auto itl=m_layer.begin()+1;itl<m_layer.end();itl++){
        for(auto &neur:*itl)
            neur.BeforNewEpoch(iternum);
        
    }
}

template <typename T>
void Net<T>::RememberOut(){
    auto it=m_layer.end()-1;
    out.clear();
    for(auto k:*it)
        out.push_back(k.m_outputVal);
}

template <typename T>
void Net<T>::addLayer(ActF<T>* ActFunc,unsigned int num_of_neurons,unsigned int out_num,double b){
    
    vector<Neuron<T>> l;
    for(int i=0;i<num_of_neurons;i++)l.push_back(Neuron<T>(ActFunc,i,out_num,b));
    m_layer.push_back(l);

}

template <typename T>
void Net<T>::addInputToLayer(vector<T> input){
    for(unsigned int i=0;i<min(input.size(),m_layer[0].size());i++){
        this->m_layer[0][i].m_outputVal=input[i];
    }
}


template <typename T>
void Net<T>::FeedForvard(){
    for(auto itl=m_layer.begin()+1;itl<m_layer.end();itl++){
        for(auto &neur:*itl)
            neur.countOut(*(itl-1));
    
    }
    RememberOut();
}


template <typename T>
void Net<T>::UpdateLayerW(vector<vector<double>> w,unsigned int LayerInd){
    for(unsigned int i=0;i<min(w.size(),m_layer[LayerInd].size());i++){
        m_layer[LayerInd][i].updateW(w[i]);
    }
}

template <typename T>
void Net<T>::UpdateAllNet(vector<vector<vector<double>>> w){
    for(unsigned int i=0;i<min(w.size(),m_layer.size());i++){
        this->UpdateLayerW(w[i], i);
    }
}


template <typename T>
void Net<T>::PrintOut() const{
    for(T k:out)
        cout<<k<<endl;
}

template <typename T>
void Net<T>::MakeNet(vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate,vector<double> bia){
    if(bia[0]==-999999){
    int z=0;
    learning_rate=rate;
    hiden_neur_num=0;
    
    for(int x=1;x<NetConf.size();x++){
        hiden_neur_num+=NetConf[x];
        
    }
    input_neurons_num=NetConf[0];
    factor_mashtab=0.7*pow(double(hiden_neur_num),1/double(input_neurons_num));
    double b;
    for(int i=0;i<NetConf.size()-1;i++){ 
        b=2*factor_mashtab*0.00001*(rand()%100000)-factor_mashtab;
        this->addLayer(functs[i],NetConf[i], NetConf[i+1],b);
        z=i;
        
    }
    b=2*factor_mashtab*0.00001*(rand()%100000)-factor_mashtab;
    this->addLayer(functs[z+1],NetConf[z+1], 0,b);
}



else {
    int z=0;
    learning_rate=rate;
    hiden_neur_num=0;
    
    for(int x=1;x<NetConf.size();x++){
        hiden_neur_num+=NetConf[x];
        
    }
    input_neurons_num=NetConf[0];
    factor_mashtab=0.7*pow(double(hiden_neur_num),1/double(input_neurons_num));
    for(int i=0;i<NetConf.size()-1;i++){
        this->addLayer(functs[i],NetConf[i], NetConf[i+1],bia[i]);
        z=i;
        
    }
    this->addLayer(functs[z+1],NetConf[z+1], 0,bia[z+1]);
}

}
template <typename T>
void Net<T>::CountErrorLastLayer(vector<T> target){
    double delta;
    auto it=m_layer.end()-1;
    auto prevL=it-1;
    for(int i=0;i<min((*it).size(),target.size());i++){
        delta=(target.at(i)-(*it)[i].m_outputVal)*((*(*it)[i].actf)[(*it)[i].sum]);
        (*it)[i].delta_bias_sum+=learning_rate*delta;
        for(unsigned int s=0;s<(*prevL).size();s++){
            (*prevL)[s].m_outputWeights[i].sum_delta+=learning_rate*delta*(*prevL)[s].m_outputVal;
            (*prevL)[s].m_outputWeights[i] .deltaWeight=delta*(*prevL)[s].m_outputVal;
        }
    }
        
}

template <typename T>
void Net<T>::CountErrorLayer(unsigned int currLayerInd){
    double delta;
    Sum<T> summ;
    auto it=m_layer.begin()+currLayerInd;
    auto prevL=m_layer.begin()+currLayerInd-1;
    for(int i=0;i<(*it).size();i++){
        summ.zero();
        summ=for_each((*it)[i].m_outputWeights.begin(),(*it)[i].m_outputWeights.end(),summ );
        delta=summ.sum*((*(*it)[i].actf)[(*it)[i].sum]);
        (*it)[i].bias+=learning_rate*delta;
        for(unsigned int s=0;s<(*prevL).size();s++){
            (*prevL)[s].m_outputWeights[i].sum_delta+=learning_rate*delta*(*prevL)[s].m_outputVal;
            (*prevL)[s].m_outputWeights[i] .deltaWeight=delta*(*prevL)[s].m_outputVal;
        }
    }
    
}

template <typename T>
void Net<T>::BackProp(vector<T> target){
    for(unsigned int i=m_layer.size()-1;i>0;i--){
        if(m_layer[i][0].typeN){
            this->CountErrorLastLayer(target);
        }
        else{
            this->CountErrorLayer(i);
        }
    }
}



template <typename T>
double Net<T>::count_accuracy(vector<vector<T>> input,vector<vector<T>> target,int resize){
    accuracy=0;
    auto it=out.begin();
    for(int j=0;j<min(target.size(),input.size())/resize ; j++ ) {
        addInputToLayer(input[j]);
        FeedForvard();
        it=max_element(out.begin(), out.end());
        if(target[j][distance(out.begin(), it)]!=0)
            accuracy++;
    }
    accuracy=accuracy/min(target.size(),input.size());
    return accuracy;
}

template <typename T>
void Net<T>::SaveWeightsAndBias(string name){
    mINI::INIFile file(name);
    mINI::INIStructure ini;
    string section,weightname;
    for(int i=0;i<this->m_layer.size();i++){
        section="layer "+to_string(i+1);
        ini[section]["bias"]=to_string(m_layer[i][0].bias);
        for(int j=0;j<m_layer[i].size();j++){
            for(int k=0;k<m_layer[i][j].m_outputWeights.size();k++){
                weightname="w"+to_string(j+1)+"_"+to_string(k+1);
                ini[section][weightname]=to_string(m_layer[i][j].m_outputWeights[k].weight);
            }
        }
    }
    file.write(ini);

}

template <typename T>
void Net<T>::Fit(vector<vector<T>> input,vector<vector<T>> target,unsigned int epoch_num,bool save,unsigned int batch_size){
    int iter;
    for(int i=0;i<epoch_num;i++){
        for(int j=0;j<min(target.size(),input.size())/batch_size ; j++ ) {
            for(int b=batch_size*j;b<batch_size*(j+1) and b<min(target.size(),input.size());b++){
            addInputToLayer(input[j]);
            FeedForvard();
            BackProp(target[j]);
                iter=b;
            }
            BeforNewEpoch(iter);
        }
        cout<<i<<endl;
    }
    if(save)SaveWeightsAndBias("Weights1");
   cout<< count_accuracy(input, target);
}
