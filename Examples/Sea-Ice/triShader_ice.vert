in vec3 aVertexPosition;
in vec3 aVertexNormal;
in vec4 aVertexColour;
in vec2 aVertexTexCoord;
flat out vec4 vFlatColour;

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

void main(void)
{
  vec3 vPos = aVertexPosition;
  vec4 mvPosition = uMVMatrix * vec4(vPos, 1.0);
  vPosEye = vec3(mvPosition) / mvPosition.w;
  gl_Position = uPMatrix * mvPosition;

  vNormal = normalize(mat3(uNMatrix) * aVertexNormal);

  if (uColour.a > 0.0)
    vColour = uColour;
  else
    vColour = aVertexColour;

  vTexCoord = aVertexTexCoord;
  vFlatColour = vColour;
  //vVertex = aVertexPosition;
  vVertex = vPos;

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
}

