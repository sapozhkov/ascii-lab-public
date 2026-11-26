/* Orthographic depth buffer for regular grids of coloured botanical tissue. */
#include <math.h>
#include <stdint.h>
#include <float.h>

typedef struct { float x,y,z,r,g,b; } Vertex;
static void triangle(Vertex a, Vertex b, Vertex c, int w, int h, float* depth, uint8_t* rgb) {
    float ux=b.x-a.x, uy=b.y-a.y, vx=c.x-a.x, vy=c.y-a.y;
    float area=ux*vy-uy*vx;
    if (fabsf(area)<0.00001f) return;
    int x0=fmaxf(0,floorf(fminf(a.x,fminf(b.x,c.x)))), y0=fmaxf(0,floorf(fminf(a.y,fminf(b.y,c.y))));
    int x1=fminf(w-1,ceilf(fmaxf(a.x,fmaxf(b.x,c.x)))), y1=fminf(h-1,ceilf(fmaxf(a.y,fmaxf(b.y,c.y))));
    for (int y=y0; y<=y1; y++) for (int x=x0; x<=x1; x++) {
        float dx=x+.5f-a.x, dy=y+.5f-a.y;
        float s=(dx*vy-dy*vx)/area, t=(ux*dy-uy*dx)/area, p=1-s-t;
        if (s<-.0001f || t<-.0001f || p<-.0001f) continue;
        int i=y*w+x;
        float z=p*a.z+s*b.z+t*c.z;
        if (z<=depth[i]) continue;
        depth[i]=z;
        rgb[i*3]=(uint8_t)(p*a.r+s*b.r+t*c.r+.5f);
        rgb[i*3+1]=(uint8_t)(p*a.g+s*b.g+t*c.g+.5f);
        rgb[i*3+2]=(uint8_t)(p*a.b+s*b.b+t*c.b+.5f);
    }
}
void grid(const Vertex* v,int rows,int cols,int w,int h,float* depth,uint8_t* rgb) {
    for (int j=0;j<rows-1;j++) for (int i=0;i<cols-1;i++){
        int k=j*cols+i;
        triangle(v[k],v[k+cols],v[k+1],w,h,depth,rgb);
        triangle(v[k+1],v[k+cols],v[k+cols+1],w,h,depth,rgb);
    }
}
/* Filaments and anthers share the petal depth buffer. Segment radii are
   expressed in screen pixels at this fixed macro scale. */
void stamens(const Vertex* base,const Vertex* tip,int count,int w,int h,float* depth,uint8_t* rgb) {
    for(int k=0;k<count;k++) {
        Vertex a=base[k], b=tip[k];
        float dx=b.x-a.x,dy=b.y-a.y,len=dx*dx+dy*dy+.001f;
        int x0=fmaxf(0,floorf(fminf(a.x,b.x)-3)),x1=fminf(w-1,ceilf(fmaxf(a.x,b.x)+3));
        int y0=fmaxf(0,floorf(fminf(a.y,b.y)-3)),y1=fminf(h-1,ceilf(fmaxf(a.y,b.y)+3));
        for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++){
            float s=fmaxf(0,fminf(1,((x+.5f-a.x)*dx+(y+.5f-a.y)*dy)/len));
            float ex=x+.5f-a.x-s*dx,ey=y+.5f-a.y-s*dy;
            float radius=s>.80f?1.35f:.67f;
            if(ex*ex+ey*ey>radius*radius)continue;
            float z=a.z+s*(b.z-a.z);
            int j=y*w+x;
            if(z<=depth[j])continue;
            depth[j]=z; float fac=.70f+.30f*s;
            rgb[3*j]=(uint8_t)(fac*b.r);rgb[3*j+1]=(uint8_t)(fac*b.g);rgb[3*j+2]=(uint8_t)(fac*b.b);
        }
    }
}
