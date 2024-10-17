__kernel void kernelMain(__read_only image2d_t inputImage, __write_only image2d_t outputImage, int rows, int cols, float tilt, float pitch, float center, float viewPortionElement, float subp, float focus) 
{
    const sampler_t imageSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    int2 coords = (int2)(get_global_id(0), get_global_id(1)); 
    float2 coordsNormalized = (float2)((float)get_global_id(0)/get_image_width(outputImage), (float)get_global_id(1)/get_image_height(outputImage));

    float newTilt = -tilt;

    int ri = 0;
    int bi = 2;
    float4 tile = (float4)(cols, rows, cols*rows, cols*rows);
    float4 viewPortion = (float4)(viewPortionElement, viewPortionElement, 0.0f, 0.0f);
  
  	float3 nuv = (float3)(coordsNormalized[0], coordsNormalized[1], 0.0f);
 
    float temp; 
  	float4 rgb[3];
  	for (int i=0; i < 3; i++) 
  	{
        nuv[2] = (coordsNormalized[0] + i * subp + coordsNormalized[1] * newTilt) * pitch - center;
        nuv[2] = 1.0f - fract(nuv[2], &temp);

        float3 uvz = nuv;
        float z = floor(uvz[2] * tile[2]);
        float focusMod = focus*(1.0f-2.0f*clamp(z/tile[2],0.0f,1.0f));
        float x = (fmod(z, tile[0]) + clamp(uvz[0]+focusMod,0.0f,1.0f)) / tile[0];
        float y = (floor(z / tile[0]) + uvz[1]) / tile[1]; 
        float2 texArr = (float2)(1.0f - x * viewPortion[0], y * viewPortion[1]);

  		rgb[i] = convert_float4(read_imageui(inputImage, imageSampler, texArr));
  	}
  
    uint4 pixel = convert_uint4(round((float4)(rgb[ri][0], rgb[1][1], rgb[bi][2], 255.0f)));
	write_imageui(outputImage, coords, pixel);
}
