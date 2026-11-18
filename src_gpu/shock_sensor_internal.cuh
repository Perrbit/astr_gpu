#pragma once

namespace astr {
__host__ __device__ inline size_t shock_node_offset(const int upper[3],const int p[3])
{
    return p[0]+static_cast<size_t>(upper[0]+1)*(p[1]+static_cast<size_t>(upper[1]+1)*p[2]);
}

__host__ __device__ inline size_t shock_faces_size(const int upper[3])
{
    size_t size=0;
    for(int a=0;a<3;++a) size+=2*static_cast<size_t>(upper[(a+1)%3]+1)*(upper[(a+2)%3]+1);
    return size;
}

__host__ __device__ inline size_t shock_face_offset(const int upper[3],int axis,int side,const int p[3])
{
    size_t base=0;
    for(int a=0;a<axis;++a) base+=2*static_cast<size_t>(upper[(a+1)%3]+1)*(upper[(a+2)%3]+1);
    const size_t n1=upper[(axis+1)%3]+1,n2=upper[(axis+2)%3]+1;
    return base+side*n1*n2+p[(axis+1)%3]+n1*p[(axis+2)%3];
}

__device__ inline bool shock_interface_active(const int8_t *mask,const int8_t *faces,
    const int upper[3],const int point[3],int axis)
{
    int left[3]={point[0],point[1],point[2]},right[3]={point[0],point[1],point[2]};
    ++right[axis];
    if(faces) {
        const int8_t l=left[axis]<0 ? faces[shock_face_offset(upper,axis,0,left)] : mask[shock_node_offset(upper,left)];
        const int8_t r=right[axis]>upper[axis] ? faces[shock_face_offset(upper,axis,1,right)] : mask[shock_node_offset(upper,right)];
        return l!=0 || r!=0;
    }
    left[axis]=max(0,point[axis]);right[axis]=min(upper[axis],point[axis]+1);
    return mask[shock_node_offset(upper,left)]!=0 || mask[shock_node_offset(upper,right)]!=0;
}
}
