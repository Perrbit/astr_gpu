#pragma once

// Existing diagonal-norm SBP 6/3 table from src/keep_sbp63_body.inc.
namespace astr { namespace sbp63 {
static __device__ __constant__ double hleft[6]={13649.0/43200,12013.0/8640,2711.0/4320,
    5359.0/4320,7877.0/8640,43801.0/43200};
static __device__ __constant__ double dleft[6][9]={
    {-1.58253351893911631,2.04489044241717277,-.187559913610859469,-.381327724279292901,
     .0584410144180559574,.0480896999940402387,0,0,0},
    {-.464675096121734577,0,.313417627884012617,.206498077565308052,
     -.0298735581739810498,-.0253670511536050539,0,0,0},
    {.0944302936508528562,-.694409067460465490,0,.722151468254264239,
     -.138818134920930952,.0166454404762792808,0,0,0},
    {.0971215172362020623,-.231448162510920485,-.365320513237042455,0,.549581945988428577,
     -.0633701298912972694,.0134353424146295957,0,0},
    {-.0202529238489791985,.0455593569054252023,.0955531201652009138,-.747799834594893631,0,
     .773188853164537782,-.164529643265202502,.0182810714739113887,0},
    {-.0149854184885882796,.0347862361028580952,-.0103024563665654063,.0775325965360293412,
     -.695236249900352044,0,.739709139060752086,-.147941827812150400,.0164379808680167119}};

__device__ inline double norm(int global_node,int total)
{
    const int distance=min(global_node,total-global_node);
    return distance<6 ? hleft[distance] : 1;
}

__device__ inline double weight(int global_node,int offset,int total)
{
    int column=global_node+offset;
    if(column<0 || column>total) return 0;
    if(global_node<6) return column<=8 ? dleft[global_node][column] : 0;
    if(total-global_node<6) {
        column=total-column;
        return column<=8 ? -dleft[total-global_node][column] : 0;
    }
    switch(offset) {
    case -3:return -1.0/60;case -2:return 3.0/20;case -1:return -3.0/4;
    case 1:return 3.0/4;case 2:return -3.0/20;case 3:return 1.0/60;
    default:return 0;
    }
}

__device__ inline double axis_weight(int axis,int globalj,int offset,int total)
{
    return axis==1 ? weight(globalj,offset,total) : weight(6,offset,12);
}
}}
