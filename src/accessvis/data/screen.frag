
in vec4 vColour;
in vec3 vVertex;
in vec2 vTexCoord; // scale 0 .. 1

//Custom uniform
uniform sampler2D uTexture;
uniform float  widthToHeight = 1; // 1 means square, 2 means width is double height
uniform float scale = 0.5; // Scale down to half the height of the box
uniform vec2 offset = vec2(0,0);

out vec4 outColour;

void main(void)
{
  vec2 size = vec2(scale*widthToHeight, scale);
  vec2 texCoord = vTexCoord/size;
  texCoord.y = 1-texCoord.y;

  texCoord.x += (1 - 1/size.x) * offset.x;
  texCoord.y += (1/size.y - 1) * offset.y;


  
  if (texCoord.x >= 0.0 && texCoord.y >= 0.0 && texCoord.x <= 1.0 && texCoord.y <= 1.0)
    outColour = texture(uTexture, texCoord);
  else
    discard;

  //Discard transparent to skip depth write
  //This fixes depth buffer output interfering with other objects
  // in transparent areas
  if (outColour.a <= 0.1)
    discard;
}

