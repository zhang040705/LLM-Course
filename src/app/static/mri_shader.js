/**
 * 全息核磁切片台 WebGL1 片元着色器背景引擎
 * 支持程序化 MRI 同心断层噪声 + AI 生成核磁纹理采样 + 4 态平滑过渡
 */

const VERTEX_SHADER_SRC = `
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER_SRC = `
precision mediump float;

varying vec2 vUv;

uniform float uTime;
uniform vec2 uResolution;
uniform sampler2D uTexture;
uniform float uHasTex;
uniform float uScanWeight;
uniform float uGroundWeight;
uniform float uRefuseWeight;
uniform float uConfidence;
uniform float uPulse;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 4; i++) {
    v += amp * noise(p);
    p *= 2.02;
    amp *= 0.5;
  }
  return v;
}

void main() {
  vec2 aspect = vec2(uResolution.x / max(uResolution.y, 1.0), 1.0);
  vec2 centered = (vUv - 0.5) * aspect;

  // 拒答伪影干扰态时叠加极轻量的水平相位偏移
  float glitchOffset = uRefuseWeight * 0.012 * sin(vUv.y * 85.0 + uTime * 22.0)
    * step(0.86, sin(uTime * 7.0 + vUv.y * 12.0));
  vec2 sampleCoord = centered + vec2(glitchOffset, 0.0);

  // 程序化 MRI 横断面同心波纹基底层
  float dist = length(sampleCoord * vec2(1.05, 0.92));
  float n = fbm(sampleCoord * 4.2 + vec2(0.0, uTime * 0.04));
  float rings = smoothstep(0.35, 0.95, sin((dist + n * 0.14) * 34.0 - uTime * 0.35) * 0.5 + 0.5);
  float coreMask = smoothstep(0.55, 0.08, dist);
  float procedural = rings * coreMask * 0.45 + coreMask * n * 0.25;

  // AI 生成核磁断层贴图采样（居中方形适应）
  float texLuminance = 0.0;
  if (uHasTex > 0.5) {
    vec2 texUv = sampleCoord * 1.15 + 0.5;
    if (texUv.x >= 0.0 && texUv.x <= 1.0 && texUv.y >= 0.0 && texUv.y <= 1.0) {
      vec3 texel = texture2D(uTexture, texUv).rgb;
      float edgeFade = smoothstep(0.0, 0.12, texUv.x) * smoothstep(1.0, 0.88, texUv.x)
        * smoothstep(0.0, 0.12, texUv.y) * smoothstep(1.0, 0.88, texUv.y);
      texLuminance = dot(texel, vec3(0.299, 0.587, 0.114)) * edgeFade;
    }
  }

  float density = mix(procedural, max(texLuminance * 0.85, procedural * 0.45), uHasTex);

  // 微弱放射科坐标系标尺与十字准星
  vec2 grid = abs(fract(sampleCoord * 6.0) - 0.5);
  float gridLine = 1.0 - smoothstep(0.0, 0.015, min(grid.x, grid.y));
  float crosshair = (1.0 - smoothstep(0.0, 0.004, abs(centered.x))) * 0.35
    + (1.0 - smoothstep(0.0, 0.004, abs(centered.y))) * 0.35;

  // 扫描光束（在 SCANNING 与脉冲期间增强）
  float scanPos = fract(uTime * 0.28);
  float scanDist = abs((1.0 - vUv.y) - scanPos);
  float scanBeam = exp(-scanDist * 38.0) * (0.18 + 0.62 * uScanWeight + 0.35 * uPulse);

  // 色谱定义（克制暗室亮度，确保前景切片可读性）
  vec3 baseBg = vec3(0.020, 0.035, 0.067);       // #050911
  vec3 idleColor = vec3(0.08, 0.26, 0.38);        // 深海蓝
  vec3 cyanColor = vec3(0.133, 0.827, 0.933);     // #22D3EE
  vec3 emeraldColor = vec3(0.204, 0.827, 0.600);  // #34D399
  vec3 amberColor = vec3(0.961, 0.620, 0.043);    // #F59E0B
  vec3 coralColor = vec3(0.984, 0.443, 0.522);    // #FB7185

  float breathe = 0.85 + 0.15 * sin(uTime * 1.4);
  vec3 groundPalette = mix(cyanColor, emeraldColor, smoothstep(0.35, 0.92, uConfidence));
  vec3 refusePalette = mix(amberColor, coralColor, 0.35 + 0.25 * sin(uTime * 5.0));

  vec3 activeTint = idleColor;
  activeTint = mix(activeTint, cyanColor, uScanWeight);
  activeTint = mix(activeTint, groundPalette, uGroundWeight);
  activeTint = mix(activeTint, refusePalette, uRefuseWeight);

  float energy = density * (0.26 + 0.24 * uGroundWeight + 0.22 * uScanWeight + 0.22 * uRefuseWeight + 0.25 * uPulse) * breathe;
  vec3 col = baseBg;
  col += activeTint * energy;
  col += activeTint * scanBeam * 0.36;
  col += activeTint * gridLine * (0.028 + 0.045 * uRefuseWeight);
  col += activeTint * crosshair * 0.045;

  // 边缘暗角收拢焦点
  float vignette = smoothstep(0.95, 0.2, length(vUv - 0.5));
  col *= (0.72 + 0.28 * vignette);

  gl_FragColor = vec4(col, 1.0);
}
`;

export class MRIShaderCanvas {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {string} textureUrl
   */
  constructor(canvas, textureUrl) {
    this.canvas = canvas;
    this.textureUrl = textureUrl;
    this.gl = null;
    this.program = null;
    this.uniforms = {};
    this.hasTex = 0.0;

    // 0: IDLE, 1: SCANNING, 2: GROUNDED, 3: REFUSED
    this.targetMode = 0;
    this.weights = { scan: 0, ground: 0, refuse: 0 };
    this.confidence = 0.0;
    this.targetConfidence = 0.0;
    this.pulse = 0.0;

    this._rafId = null;
    this._lastTs = performance.now();
    this._startTs = this._lastTs;
    this._resizeObserver = null;

    this._init();
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.warn("MRI Shader compile warning:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  _init() {
    const gl = this.canvas.getContext("webgl", {
      alpha: false,
      antialias: false,
      powerPreference: "high-performance",
    });
    if (!gl) {
      this.canvas.classList.add("webgl-fallback");
      return;
    }
    this.gl = gl;

    const vs = this._compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
    if (!vs || !fs) {
      this.canvas.classList.add("webgl-fallback");
      return;
    }

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      this.canvas.classList.add("webgl-fallback");
      return;
    }
    this.program = prog;
    gl.useProgram(prog);

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(prog, "aPosition");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const names = [
      "uTime",
      "uResolution",
      "uTexture",
      "uHasTex",
      "uScanWeight",
      "uGroundWeight",
      "uRefuseWeight",
      "uConfidence",
      "uPulse",
    ];
    for (const name of names) {
      this.uniforms[name] = gl.getUniformLocation(prog, name);
    }

    this._loadTexture(this.textureUrl);
    this._setupResize();
    this._loop(performance.now());
  }

  _loadTexture(url) {
    if (!this.gl || !url) return;
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // 1x1 占位像素避免渲染警告
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255])
    );

    const img = new Image();
    img.onload = () => {
      if (!this.gl) return;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      this.hasTex = 1.0;
    };
    img.src = url;
  }

  _resize() {
    if (!this.gl) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.0);
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
  }

  _setupResize() {
    this._resize();
    if (typeof ResizeObserver !== "undefined") {
      this._resizeObserver = new ResizeObserver(() => this._resize());
      this._resizeObserver.observe(this.canvas);
    } else {
      window.addEventListener("resize", () => this._resize());
    }
  }

  /**
   * 切换着色器工作模式
   * @param {"idle"|"scanning"|"grounded"|"refused"} mode
   * @param {number} confidence
   */
  setMode(mode, confidence = 0.0) {
    const map = { idle: 0, scanning: 1, grounded: 2, refused: 3 };
    this.targetMode = map[mode] ?? 0;
    this.targetConfidence = Math.max(0.0, Math.min(1.0, Number(confidence) || 0.0));
  }

  triggerPulse(amount = 0.45) {
    this.pulse = Math.min(1.0, this.pulse + amount);
  }

  _loop(now) {
    if (!this.gl || !this.program) return;
    const dt = Math.min(0.1, Math.max(0.001, (now - this._lastTs) / 1000.0));
    this._lastTs = now;
    const elapsed = (now - this._startTs) / 1000.0;

    const k = 1.0 - Math.exp(-dt * 4.5);
    const tScan = this.targetMode === 1 ? 1.0 : 0.0;
    const tGround = this.targetMode === 2 ? 1.0 : 0.0;
    const tRefuse = this.targetMode === 3 ? 1.0 : 0.0;

    this.weights.scan += (tScan - this.weights.scan) * k;
    this.weights.ground += (tGround - this.weights.ground) * k;
    this.weights.refuse += (tRefuse - this.weights.refuse) * k;
    this.confidence += (this.targetConfidence - this.confidence) * k;
    this.pulse *= Math.exp(-dt * 4.2);

    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform1f(this.uniforms.uTime, elapsed);
    gl.uniform2f(this.uniforms.uResolution, this.canvas.width, this.canvas.height);
    gl.uniform1i(this.uniforms.uTexture, 0);
    gl.uniform1f(this.uniforms.uHasTex, this.hasTex);
    gl.uniform1f(this.uniforms.uScanWeight, this.weights.scan);
    gl.uniform1f(this.uniforms.uGroundWeight, this.weights.ground);
    gl.uniform1f(this.uniforms.uRefuseWeight, this.weights.refuse);
    gl.uniform1f(this.uniforms.uConfidence, this.confidence);
    gl.uniform1f(this.uniforms.uPulse, this.pulse);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    this._rafId = requestAnimationFrame((ts) => this._loop(ts));
  }
}
