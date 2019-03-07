//
//  Builder.cpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 16/02/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//

#include "Builder.hpp"

template<typename T>
ActF<T> *   StrToFunc(string name,ActFContainer<T>* functions){
    if (name=="linear") return &(functions->linear);
    else if(name=="sigm") return &(functions->sigm);
    else if(name=="bipsigm") return &(functions->bigsim);
    else if(name=="purelin") return &(functions->purelin);
    else if(name=="tansig") return &(functions->tansig);
    else if(name=="tg") return &(functions->tg);
    else return &functions->linear;
    
}

template<typename T>
void GdBuilder<T>::createNet(vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate) {
    GdBuilder<T>::net->MakeNet(functs,NetConf,rate);
    }

template<typename T>
void GdBuilder<T>::UpdateWeights(vector<vector<vector<double>>> w) {
    this->net->UpdateAllNet(w);
    }

template<typename T>
void GdBuilder<T>::LoadWeightsFromIni(const string name){
    
    mINI::INIFile file(name);
    mINI::INIStructure ini;
    file.read(ini);
    string section,weightname;
    for(int i=0;i<this->net->m_layer.size();i++){
        section="layer "+to_string(i+1);
        for(int j=0;j<this->net->m_layer[i].size();j++){
            this->net->m_layer[i][j].bias=atof(ini[section]["bias"].c_str());
            for(int k=0;k<this->net->m_layer[i][j].m_outputWeights.size();k++){
                weightname="w"+to_string(j+1)+"_"+to_string(k+1);
                    this->net->m_layer[i][j].m_outputWeights[k].weight=atof(ini[section][weightname].c_str());
            }
        }
    }

}
/*
template<typename T>
void GdBuilder<T>::LoadWeightsFromBinaryOneLayer(const string name,unsigned int l_num){
    vector<vector<double> > w1(this->net->m_layer[l_num].size(), vector<double>(this->net->m_layer[l_num][0].m_outputWeights.size(), 0));
    ifstream in;
    in.open("/Users/valyagorbunova/Library/Developer/Xcode/DerivedData/NeuralNetwork-gtsewaeridktwjbexvbrpkbvjuix/Build/Products/Debug/net_train_weight2.bin", ios::in|ios::binary);
    double d;
    if(!in.is_open()) abort();
   double w [this->net->m_layer[l_num].size()][this->net->m_layer[l_num][0].m_outputWeights.size()];
    for(int i=0;i<this->net->m_layer[l_num].size();i++){
    in.read((char*)w[i],sizeof(double)*this->net->m_layer[l_num][0].m_outputWeights.size());
    }
    for(int i=0;i<this->net->m_layer[l_num].size();i++){
        for(int j=0;j<)
    }
}
*/
template<typename T>
Net<T>& Director<T>::createWithRandomWeightsFromFunc(NetBuilder<T> & builder,vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate)
{
    builder.createNet(functs,NetConf,rate);
    return( *builder.net);
    }

template<typename T>
Net<T>& Director<T>::createWithWeightsFromFunc(NetBuilder<T> & builder,vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate,vector<vector<vector<double>>> w)
{
    builder.createNet(functs,NetConf,rate);
    builder.UpdateWeights(w);
    return( *builder.net);
    }




template<typename T>
Net<T>& Director<T>::createWithRandomWeightsFromINI(NetBuilder<T> & builder,string name){
    mINI::INIFile file(name);
    mINI::INIStructure ini;
    file.read(ini);
    string section;
    vector<ActF<T> *> functs;
    vector<unsigned int> NetConf;
    double rate=atof(ini["meta"]["rate"].c_str());
    unsigned int l_num=atoi(ini["meta"]["l_num"].c_str());
    for(int i=0;i<l_num;i++){
        section="layer "+to_string(i+1);
        NetConf.push_back(atoi(ini[section]["neuron_num"].c_str()));
        functs.push_back(StrToFunc(ini[section]["act_func"],builder.functions));
    }
    builder.createNet(functs,NetConf,rate);
    
    return *builder.net;
}

template<typename T>
Net<T>& Director<T>::createWithWeightsFromINI(NetBuilder<T> & builder,string nameConf,string nameW){
    createWithRandomWeightsFromINI(builder,nameConf);
    builder.LoadWeightsFromIni(nameW);
    return *builder.net;
}
