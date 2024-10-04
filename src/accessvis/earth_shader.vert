in vec3 aVertexPosition;
in vec3 aVertexNormal;
in vec4 aVertexColour;
in vec2 aVertexTexCoord;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat4 uNMatrix;

uniform vec4 uColour;
uniform vec4 uLightPos;

out vec4 vColour;
out vec3 vNormal;
out vec3 vPosEye;
out vec2 vTexCoord;
out vec3 vVertex;
out vec3 vLightPos;

out mat3 TBN;

uniform float uTime;
uniform int uFrame;

//Custom 
uniform bool bathymetry = false;
uniform float radius;
uniform sampler2D uTexture;
uniform sampler2D wavetex;

#define PI 3.14159265

float rand(vec2 co)
{
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main(void)
{
  vec4 mvPosition = uMVMatrix * vec4(aVertexPosition, 1.0);

  if (uColour.a > 0.0)
    vColour = uColour;
  else
    vColour = aVertexColour;

  vTexCoord = aVertexTexCoord;
  vVertex = aVertexPosition;

  //Head light, lightPos=(0,0,0) - vPosEye
  //vec3 lightDir = normalize(uLightPos.xyz - vPosEye);
  if (uLightPos.w < 0.5)
  {
    //Light follows camera - default mode
    vLightPos = uLightPos.xyz;
  }
  else
  {
    //Fixed Scene Light, when lightpos.w set to 1.0
    vLightPos = (uMVMatrix * vec4(uLightPos.xyz, 1.0)).xyz;
  }

  //Flatten water coloured vertices to avoid coastline artifacts
  //where topo and relief texture don't align perfectly
  vec3 vertexNormal = aVertexNormal;
  //Need to lookup the texture to get relief colour
  bool water = false;
  if (vTexCoord.x > -1.0) //Null texcoord (-1,-1)
  {
    vColour = texture(uTexture, vTexCoord);
    //Land/water mask is baked into alpha channel
    float mask = vColour.a;
    water = mask < 1.0;
  }

  if (water && !bathymetry) // || vWaterlevel > 0.000005)
  {
    //Flatten vertex and normal (replace with sphere normal)
    vec3 V = normalize(aVertexPosition);
    vec3 N = normalize(mat3(uNMatrix) * V); //P / radius;

    //N = normalize(aVertexNormal); //Or just use normalize(P) as it's a sphere
    vec3 P = aVertexPosition; //mvPosition.xyz; //aVertexPosition;
    //float radius = length(P);
    V *= radius;
    mvPosition = uMVMatrix * vec4(V, 1.0);
    //vVertex = V;

    //Calc tangent
    //float theta = asin(P.y / radius); //[-pi/2,pi/2]
    //float phi = atan(P.z, P.x); //[-pi,+pi]
    //N = normalize(mat3(uNMatrix) * vertexNormal); //P / radius;
    //vec3 T = vec3(-sin(phi), 0, cos(phi));

    //Faster equivalent shortcut
    vec3 A = vec3(0, radius, 0);
    vec3 C = vec3(0,0,0); //Centre
    vec3 T = normalize(cross(A, P - C));
    vec3 B = cross(N,T);
    TBN = mat3(T, B, N);
    //vColour.rgb = B;
    //vColour.a = 1.0;
    vNormal = N;
  }
  else
  {
    //Use normal from topo
    vNormal = normalize(mat3(uNMatrix) * normalize(aVertexNormal));
  }

  vPosEye = vec3(mvPosition) / mvPosition.w;
  gl_Position = uPMatrix * mvPosition;

}

