in vec4 vColour;
in vec3 vNormal;
in vec3 vPosEye;
in vec3 vVertex;
in vec2 vTexCoord;
in vec3 vLightPos;

uniform float uOpacity;
uniform bool uLighting;
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;
uniform float uAmbient;
uniform float uDiffuse;
uniform float uSpecular;
uniform float uShininess;

uniform bool uTextured;
uniform sampler2D uTexture;
uniform vec3 uClipMin;
uniform vec3 uClipMax;
uniform bool uOpaque;
uniform vec4 uLight;

uniform float uTime;
uniform int uFrame;
uniform vec4 uViewport;

//Custom 
uniform float radius;
uniform sampler2D wavetex;
uniform sampler2D wavenormal;
uniform sampler2D blendTex;
uniform float blendFactor = -1.0;
uniform sampler2D data;
uniform int dataMode = -1; //-1 = don't plot, 0 = plot everywhere, 1 = plot on ocean, 2 = plot on land
uniform float dataAlpha = 0.0;
uniform vec4 ocean = vec4(0.0, 0.0, 0.0, 1.0); //Ocean colour override
uniform float depthColour = 2.0; //Power of bathymetry depth on ocean colour, 0=None, 1=linear, 2=^2 etc
uniform bool waves = false;
uniform bool bathymetry = false;
//Topo/bathy range
uniform float heightmin;
uniform float heightmax;

//Allow differing brightness,contrast,saturation over ocean
uniform float ocean_brightness = 0.0;
uniform float ocean_contrast = 0.0;
uniform float ocean_saturation = 0.0;
uniform bool bluemarble; //Enabled when using blue marble textures

#define isnan3(v) any(isnan(v))
out vec4 outColour;

uniform bool uCalcNormal;

in mat3 TBN;

void calcColour(vec3 colour, float alpha, float brightness, float saturation, float contrast)
{
  //Brightness adjust
  colour += brightness;
  //Saturation & Contrast adjust
  const vec3 LumCoeff = vec3(0.2125, 0.7154, 0.0721);
  vec3 AvgLumin = vec3(0.5, 0.5, 0.5);
  vec3 intensity = vec3(dot(colour, LumCoeff));
  colour = mix(intensity, colour, saturation);
  colour = mix(AvgLumin, colour, contrast);

  //Gamma correction
  //outColour = vec4(pow(colour, vec3(1.0 / uGamma)), alpha);

  outColour = vec4(colour, alpha);
}

float rand(vec2 co)
{
  return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}
uniform mat4 uNMatrix;

// All components are in the range [0…1], including hue.
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

void main(void)
{
  float alpha = 1.0; //fColour.a;
float mask = 0.0;
vec4 tColour = texture(uTexture, vTexCoord);
mask = tColour.a;

if (blendFactor >= 0.0)
{
  //Blend between two textures
  vec4 tColour2 = texture(blendTex, vTexCoord);
  tColour = mix(tColour, tColour2, blendFactor);
}

  vec4 fColour = tColour;

  vec3 lightColour = uLight.xyz;
  
  //Light direction
  vec3 lightDir = normalize(vLightPos - vPosEye);

  //Calculate diffuse lighting
  vec3 N = normalize(vNormal);

  //Modified to use energy conservation adjustment
  //https://learnopengl.com/Advanced-Lighting/Advanced-Lighting
  const float kPi8 = 3.14159265 * 8.0;

  //Custom - water blend
  //Compare vertex length to water height
  float specular = uSpecular;
  float shininess = uShininess;

  float brightness = uBrightness;
  float saturation = uSaturation;
  float contrast = uContrast;
  float diff = uDiffuse;

  vec3 hsv = rgb2hsv(fColour.xyz);
  //if (hsv.x > 230.0/360.0 && hsv.x < 280.0 / 360.0) // && hsv.z > 0.35 && hsv.z < 0.65)
  //if ((fColour.z > 0.5 && fColour.x < 0.3 && fColour.y > 0.3)) // || (fColour.x * fColour.y * fColour.z > 0.25))
  
  //Ocean/land mask, baked into colour texture alpha channel
  bool water = mask <= 0.7;
  //bool water = mask >= 0.85 && depth <= 1.0;
  //Water calc for shaded relief textures
  //bool water = (fColour.z > 0.5 && fColour.x < 0.3 && fColour.y > 0.3);
  //bool snow = (fColour.x * fColour.y * fColour.z > 0.5);
  bool snow = hsv.y < 0.1 && hsv.z > 0.5;

  //Detect blank areas in land texture that should be flagged as water
  //(Fixes artifacts on edges of antarctica)
  float vlen = length(vVertex);
  //water = water || (!snow && hsv.z < 0.09 && (vlen - radius < 0.001));
  //if (!snow && hsv.z < 0.09 && (vlen - radius < 0.001))
  //Limit to southerly regions ... TODO: also fix artifacts around ice in north polar regions
  if (!water && vVertex.y < -0.85*vlen && !snow && hsv.z < 0.19 && (vlen - radius < 0.01))
  {
    water = true;
    //N = normalize(vVertex);
    //Flatten normal
    N = normalize(mat3(uNMatrix) * normalize(vVertex));
  }

  if (water && !snow) //Sea-level ice fix, don't apply water shading
  {
    //Bathymetry is included in topo data
    //Normalised depth [0,1] where 0 is sea level
    float depth = ((vlen - radius) / (heightmin));

    //Calculate the ocean colour
    //if (bluemarble)
    if (!bathymetry)
    {
      //blend bathymetry with earth texture based on depth to 
      //show sense of water depth and some blurred bathymetry detail

      //Remove detail of lower depths by clamping to upper range
      float depth2 = clamp(pow(1.0-depth, depthColour), 0.6, 1.0) - 0.4;
      //float depth2 = pow(1.0-depth, 2*depthColour);

      //Default ocean colour, not used unless explicitly set
      vec3 c3 = ocean.rgb;
      float blend = ocean.a;
      if (blend == 0.0 || ocean.rgb == vec3(0.0, 0.0, 0.0))
      {
        //Ignore flat colour
        c3 = vec3(40/255.0, 0.4 + min(depth2, 0.6), 255/255.0);
        blend = 0.2 * depth2;
        if (!waves)
          blend = 0.15 * depth2;
      }
      else
      {
        //Blend bathymetry with fixed ocean colour
        c3 = mix(fColour.rgb*depth2, ocean.rgb, depth2);
      }

      fColour.rgb = mix(fColour.rgb, c3, blend);
    }
    else
    {
      //Flat ocean colour
      float depth2 = pow(1.0-depth, depthColour);
      vec3 c3 = ocean.rgb * pow(1.0-depth, depthColour);
      float blend = clamp(ocean.a*sqrt(depth), 0.0, 1.0); //1.0; //depth; //ocean.a * depth;
      fColour.rgb = mix(fColour.rgb, c3, blend);
    }

    //Plotting ocean as surface
    if (!bathymetry && waves)
    {
      //Apply ocean texture
      if (waves)
      {
        //Repeated tiling using normal as texcoord
        //    vec3 NN = normalize(vPosEye); //normalize(vVertex);
        //vec4 q = vec4(0, 0.7071068, 0, 0.7071068); //90 deg about Y axis
        //vec3 temp = cross(q.xyz, NN) + q.w * NN;
        //vec3 rotated = NN + 2.0*cross(q.xyz, temp);
        //   NN = rotated;

        //NN = 3D coord on the unit sphere
        //    float latitude = acos(NN.z);
        //    float longitude = atan(NN.x, NN.y);
        vec3 P = N; //vNormal; //vVertex; //vec3(0.0, 0.0, radius);
        float longitude = asin(P.y); //[-pi/2,pi/2]
        float latitude = atan(P.z, P.x); //[-pi,+pi]
            vec2 uv = vec2(latitude, longitude);

        //vec2 uv = vTexCoord;
        //Repeated tiling
        uv = fract(uv * 25.0 + float(uFrame) * 0.005);
        vec4 wavetex = texture(wavetex, uv);
        //Apply normal map in tangent space with TBN matrix
        //vec3 waveN = texture(wavenormal, uv).xyz;
        vec3 waveN = texture(wavenormal, uv).xyz;
        //Convert to tangent space
        vec3 N3 = normalize(waveN * 2.0 - 1.0);
        N3 = normalize(TBN * N3);

        //Wave texture shading : colour effect only, looks good in shallower waters
        //fColour.rgb = fColour.rgb * pow(wavetex.r, 0.4) + 0.1;
        //fColour.rgb = fColour.rgb * pow(wavetex.r, 0.8);
        fColour.rgb = fColour.rgb * pow(wavetex.r, 0.75);
        //fColour.rgb = vec3(uv.x, uv.y, 0.0);
        //outColour.rgb = fColour.rgb;
        //outColour.a = 1.0;
        //return;

        //Wave normal map - fade in as depth increases
        //Only start adding wave normal away from land edges or we get artifacts
        if (mask < 0.55)
        {
          //As depth increases, show higher waves
          float wavesize = 0.5 * (depth);
          //N = mix(N3, N, min(1.25*depth, 1.0));
          N = normalize(mix(N, N3, wavesize));
          //N = mix(N3, N, depth);
          //fColour.rgb = N;
        }
      }

      //Enhance ocean when using darker texture
      if (bluemarble)
      {
        float mul = waves ? 1.0 : 0.5;
        saturation = 1.0 + pow(mul*clamp(1.0-depth, 0.0, 1.0), 3.0);
        contrast *= 1.05;
      }

      brightness = ocean_brightness > 0.0 ? ocean_brightness : brightness;
      contrast = ocean_contrast > 0.0 ? ocean_contrast : contrast;
      saturation = ocean_saturation > 0.0 ? ocean_saturation : saturation;
    }
  }

#define PI 3.1415926
  //Blend in data texture
  if (dataMode >= 0)
  {
    if (dataMode == 0 || (dataMode == 1 && water) || (dataMode == 2 && !water))
    {
        // radius, theta, phi

    //float theta = acos(vVertex.z/radius);
    //float phi   = atan(vVertex.y/vVertex.x);

  //NN = 3D coord on the unit sphere
      //vec3 NN = normalize(vPosEye);
      vec3 NN = normalize(vVertex);
      float latitude = acos(NN.y);
      float longitude = atan(NN.x, NN.z);
  //vec3 P = normalize(vVertex); //N; //vNormal; //vVertex; //vec3(0.0, 0.0, radius);
  //float longitude = asin(P.y); //[-pi/2,pi/2]
  //float latitude = atan(P.z, P.x); //[-pi,+pi]
      vec2 uv = vec2(longitude, latitude);

    uv.x = clamp(0.5 + uv.x/(2.0*PI), 0.0, 1.0);
    uv.x = fract(uv.x + 0.5); //Rotate 180, left edge of texture is at prime meridian
    uv.y = clamp(uv.y/PI, 0.0, 1.0);
      //vec4 dColour = texture(data, uv);
      vec4 dColour;
      //https://stackoverflow.com/questions/10564573/glsl-procedural-repetitive-texturecoordinates-cause-visible-seams-due-to-mipmapp
      if (uv.x > 0.99 || uv.x < 0.01)
        dColour = textureLod(data, uv, 0); //Without this we get seams due to mipmapping
      else
        dColour = texture(data, uv);
      //vec4 dColour = textureLod(data, uv, 0); //Without this we get seams due to mipmapping
      //fColour.rgb = vec3(uv.x, uv.y, 0.0); //mix(fColour.rgb, dColour.rgb, dColour.a);
      float a = dataAlpha;
      if (a == 0.0) a = dColour.a;
      fColour.rgb = mix(fColour.rgb, dColour.rgb, a);
      //fColour = vec4(dFdx(uv.x), dFdy(uv.y),0,1);

      //Disable specular on data plot
      if (a > 0.0)
      {
        specular = mix(specular, 0.0, a);
        //Remove any brightness/saturation/contrast adjust
        saturation = mix(saturation, 1.0, a);
        brightness = mix(brightness, 0.0, a);
        contrast = mix(contrast, 1.0, a);
      }
    }
  }


  //Calculate diffuse component
  //(Single sided lighting only)
  float diffuse = max(dot(N, lightDir), 0.0);

  if (water && !bathymetry) // && waves)
  {
    //Increase specular highlights over water
    //TODO: uniform variable for this factor
    specular *= 2.0; //0.55
    shininess *= 2.0;
    //shininess = specular * 0.2;
  }

  //Snow looks overexposed - reduce lighting
  //(interferes with texture blending so skip when enabled)
  //if (snow && uSpecular > 0.0)
  if (snow && specular > 0.5 && blendFactor < 0.0)
  {
    specular = 0.1;
    shininess = 0.05;
    diffuse = 0.6;
  }

  //Compute the specular term

  //Specular power, higher is more focused/shiny
  shininess = 256.0 * clamp(shininess, 0.0, 1.0);
  vec3 specolour = lightColour; //Color of light - use the same as diffuse/ambient
  //Blinn-Phong
  vec3 viewDir = normalize(-vPosEye);
  //Normalize the half-vector
  vec3 halfVector = normalize(lightDir + viewDir);

  //Use the adjusted normal for the specular component
  //Compute cosine (dot product) with the normal
  float NdotHV = dot(N, halfVector);
  //Single sided lighting
  NdotHV = max(NdotHV, 0.0);

  //Energy conservation adjustment (more focused/shiny highlight will be brighter)
  float energyConservation = ( 8.0 + shininess) / kPi8;
  //Multiplying specular by diffuse prevents bands at edges for low shininess
  //float spec = diffuse * specular * energyConservation * pow(NdotHV, shininess);
  float spec = specular * energyConservation * pow(NdotHV, shininess);

  //Final colour - specular + diffuse + ambient
  calcColour(lightColour * (fColour.rgb * (uAmbient + diff * diffuse) + vec3(spec)), alpha, brightness, saturation, contrast);
}

