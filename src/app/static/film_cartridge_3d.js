/**
 * Three.js 3D 拟物医疗胶片暗盒组件（Leica / Optical Medical Cartridge Dispenser）
 * 特性：倒角拉丝铝合金机身、哑光防滑侧握把、可旋转卷片旋钮、可见金属出片滚轴、物理指示灯与 3D 舌片吐片动效
 */

import * as THREE from "/static/vendor/three.module.min.js";

function createBrushedMetalTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#c8c4bc";
  ctx.fillRect(0, 0, 256, 256);

  for (let y = 0; y < 256; y += 1) {
    const v = 185 + Math.floor(Math.random() * 35);
    ctx.fillStyle = `rgb(${v},${v - 2},${v - 6})`;
    ctx.fillRect(0, y, 256, 1);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);
  return tex;
}

function createRoundedRectShape(width, height, radius) {
  const x = -width / 2;
  const y = -height / 2;
  const r = Math.min(radius, width / 2, height / 2);
  const shape = new THREE.Shape();
  shape.moveTo(x + r, y);
  shape.lineTo(x + width - r, y);
  shape.quadraticCurveTo(x + width, y, x + width, y + r);
  shape.lineTo(x + width, y + height - r);
  shape.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  shape.lineTo(x + r, y + height);
  shape.quadraticCurveTo(x, y + height, x, y + height - r);
  shape.lineTo(x, y + r);
  shape.quadraticCurveTo(x, y, x + r, y);
  return shape;
}

export class FilmCartridge3D {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.state = "idle"; // idle | scanning | ejecting | refused
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.cartridgeGroup = null;
    this.knobGroup = null;
    this.rollerFront = null;
    this.rollerBack = null;
    this.ledMat = null;
    this.ledLight = null;
    this.filmTongue = null;

    this.pointer = { x: 0, y: 0 };
    this.rotSmooth = { x: 0.14, y: -0.18 };
    this.ejectAnim = 0.0;
    this.shakeTimer = 0.0;
    this.pulseAmount = 0.0;

    this._lastTs = performance.now();
    this._startTs = this._lastTs;
    this._resizeObserver = null;

    this._init();
  }

  _init() {
    if (!this.canvas) return;
    try {
      this.renderer = new THREE.WebGLRenderer({
        canvas: this.canvas,
        alpha: true,
        antialias: true,
        powerPreference: "high-performance",
      });
    } catch (err) {
      this.canvas.classList.add("webgl-fallback");
      return;
    }

    this.renderer.setClearColor(0x000000, 0);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1.5, 0.1, 100);
    this.camera.position.set(0, 0.45, 7.2);
    this.camera.lookAt(0, -0.08, 0);

    this._setupLights();
    this._buildCartridge();
    this._setupEvents();
    this._resize();
    this._loop(performance.now());
  }

  _setupLights() {
    const hemi = new THREE.HemisphereLight(0xfffdf8, 0xd6d1c7, 1.45);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xfffaf0, 2.25);
    key.position.set(3.5, 5.5, 4.8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 18;
    key.shadow.camera.left = -4;
    key.shadow.camera.right = 4;
    key.shadow.camera.top = 4;
    key.shadow.camera.bottom = -4;
    key.shadow.bias = -0.0005;
    this.scene.add(key);

    const rimCyan = new THREE.DirectionalLight(0x99f6e4, 0.95);
    rimCyan.position.set(-4.5, 2.0, -3.2);
    this.scene.add(rimCyan);

    const fillWarm = new THREE.DirectionalLight(0xfef3c7, 0.65);
    fillWarm.position.set(-2.5, -2.0, 3.5);
    this.scene.add(fillWarm);

    const shadowPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(14, 10),
      new THREE.ShadowMaterial({ opacity: 0.14 })
    );
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -1.42;
    shadowPlane.receiveShadow = true;
    this.scene.add(shadowPlane);
  }

  _buildCartridge() {
    const brushedTex = createBrushedMetalTexture();
    const root = new THREE.Group();
    this.cartridgeGroup = root;

    // 1. 倒角拉丝铝合金主机身
    const aluminumMat = new THREE.MeshStandardMaterial({
      color: 0xe6e2da,
      roughness: 0.34,
      metalness: 0.62,
      bumpMap: brushedTex,
      bumpScale: 0.008,
    });
    const bodyShape = createRoundedRectShape(3.5, 1.38, 0.2);
    const bodyGeo = new THREE.ExtrudeGeometry(bodyShape, {
      depth: 1.56,
      bevelEnabled: true,
      bevelThickness: 0.11,
      bevelSize: 0.09,
      bevelSegments: 5,
      curveSegments: 16,
    });
    bodyGeo.center();
    const bodyMesh = new THREE.Mesh(bodyGeo, aluminumMat);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;
    root.add(bodyMesh);

    // 2. 顶部精密深灰装饰压板与光学铭牌条
    const topPlateMat = new THREE.MeshStandardMaterial({
      color: 0x32363d,
      roughness: 0.55,
      metalness: 0.35,
    });
    const plateShape = createRoundedRectShape(2.65, 1.15, 0.1);
    const plateGeo = new THREE.ExtrudeGeometry(plateShape, {
      depth: 0.08,
      bevelEnabled: true,
      bevelThickness: 0.02,
      bevelSize: 0.02,
      bevelSegments: 2,
    });
    plateGeo.center();
    const topPlate = new THREE.Mesh(plateGeo, topPlateMat);
    topPlate.rotation.x = Math.PI / 2;
    topPlate.position.set(-0.08, 0.81, 0.0);
    topPlate.castShadow = true;
    root.add(topPlate);

    const badgeMat = new THREE.MeshStandardMaterial({
      color: 0x0f766e,
      roughness: 0.25,
      metalness: 0.5,
    });
    const badge = new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.03, 0.18), badgeMat);
    badge.position.set(-0.75, 0.86, 0.32);
    root.add(badge);

    // 3. 左右侧哑光炭黑防滑橡胶护片
    const gripMat = new THREE.MeshStandardMaterial({
      color: 0x24272c,
      roughness: 0.82,
      metalness: 0.08,
    });
    const gripGeo = new THREE.BoxGeometry(0.14, 1.04, 1.38);
    const leftGrip = new THREE.Mesh(gripGeo, gripMat);
    leftGrip.position.set(-1.82, -0.02, 0);
    leftGrip.castShadow = true;
    root.add(leftGrip);

    const rightGrip = new THREE.Mesh(gripGeo, gripMat);
    rightGrip.position.set(1.82, -0.02, 0);
    rightGrip.castShadow = true;
    root.add(rightGrip);

    // 4. 右侧机械卷片旋钮
    const knobGroup = new THREE.Group();
    knobGroup.position.set(1.96, 0.26, -0.15);
    knobGroup.rotation.z = -Math.PI / 2;
    const chromeMat = new THREE.MeshStandardMaterial({
      color: 0xd4d4d8,
      roughness: 0.2,
      metalness: 0.88,
    });
    const knobBase = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.28, 0.24, 24), chromeMat);
    knobBase.castShadow = true;
    knobGroup.add(knobBase);

    for (let i = 0; i < 8; i += 1) {
      const angle = (i / 8) * Math.PI * 2;
      const rib = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.2, 0.04), topPlateMat);
      rib.position.set(Math.cos(angle) * 0.26, 0.01, Math.sin(angle) * 0.26);
      rib.rotation.y = -angle;
      knobGroup.add(rib);
    }
    this.knobGroup = knobGroup;
    root.add(knobGroup);

    // 5. 前下方出片狭缝与镀铬金属滚轴
    const slotFrameMat = new THREE.MeshStandardMaterial({
      color: 0x181c22,
      roughness: 0.65,
      metalness: 0.3,
    });
    const slotHousing = new THREE.Mesh(new THREE.BoxGeometry(2.74, 0.26, 0.24), slotFrameMat);
    slotHousing.position.set(0, -0.44, 0.82);
    root.add(slotHousing);

    const slotMouth = new THREE.Mesh(
      new THREE.BoxGeometry(2.54, 0.11, 0.14),
      new THREE.MeshBasicMaterial({ color: 0x05070a })
    );
    slotMouth.position.set(0, -0.44, 0.91);
    root.add(slotMouth);

    this.rollerFront = this._createRoller(chromeMat, topPlateMat);
    this.rollerFront.position.set(0, -0.37, 0.88);
    root.add(this.rollerFront);

    this.rollerBack = this._createRoller(chromeMat, topPlateMat);
    this.rollerBack.position.set(0, -0.51, 0.88);
    root.add(this.rollerBack);

    // 6. 前左物理玻璃指示灯珠
    this.ledMat = new THREE.MeshPhysicalMaterial({
      color: 0xfafaf9,
      emissive: 0x94a3b8,
      emissiveIntensity: 0.7,
      roughness: 0.12,
      metalness: 0.0,
      transmission: 0.35,
    });
    const ledDome = new THREE.Mesh(new THREE.SphereGeometry(0.085, 20, 16), this.ledMat);
    ledDome.position.set(-1.32, 0.36, 0.86);
    root.add(ledDome);

    const bezel = new THREE.Mesh(new THREE.TorusGeometry(0.095, 0.018, 12, 24), chromeMat);
    bezel.position.set(-1.32, 0.36, 0.86);
    root.add(bezel);

    this.ledLight = new THREE.PointLight(0x94a3b8, 0.6, 2.2);
    this.ledLight.position.set(-1.32, 0.36, 1.05);
    root.add(this.ledLight);

    // 7. 3D 物理出片舌片
    const filmMat = new THREE.MeshPhysicalMaterial({
      color: 0x0f2530,
      emissive: 0x0f766e,
      emissiveIntensity: 0.18,
      roughness: 0.22,
      metalness: 0.05,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
    });
    const tongueGeo = new THREE.BoxGeometry(2.38, 1.25, 0.025);
    this.filmTongue = new THREE.Mesh(tongueGeo, filmMat);
    this.filmTongue.position.set(0, -0.44, 0.88);
    this.filmTongue.visible = false;
    root.add(this.filmTongue);

    root.position.set(0, 0.16, 0);
    this.scene.add(root);
  }

  _createRoller(chromeMat, ringMat) {
    const grp = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 2.48, 18), chromeMat);
    shaft.rotation.z = Math.PI / 2;
    grp.add(shaft);

    for (let i = 0; i < 4; i += 1) {
      const angle = (i / 4) * Math.PI * 2;
      const strip = new THREE.Mesh(new THREE.BoxGeometry(2.44, 0.012, 0.012), ringMat);
      strip.position.set(0, Math.cos(angle) * 0.041, Math.sin(angle) * 0.041);
      grp.add(strip);
    }
    return grp;
  }

  _setupEvents() {
    window.addEventListener("pointermove", (e) => {
      const nx = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
      const ny = (e.clientY / Math.max(1, window.innerHeight)) * 2 - 1;
      this.pointer.x = nx;
      this.pointer.y = ny;
    });

    if (typeof ResizeObserver !== "undefined") {
      this._resizeObserver = new ResizeObserver(() => this._resize());
      this._resizeObserver.observe(this.canvas);
    } else {
      window.addEventListener("resize", () => this._resize());
    }
  }

  _resize() {
    if (!this.renderer || !this.camera) return;
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(240, Math.floor(rect.width));
    const height = Math.max(160, Math.floor(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2.0);

    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  resizeNow() {
    this._resize();
  }

  /**
   * @param {"idle"|"scanning"|"ejecting"|"refused"} nextState
   */
  setState(nextState) {
    this.state = nextState;
    if (nextState === "refused") {
      this.shakeTimer = 0.55;
      this.ejectAnim = 1.0;
    }
  }

  triggerEject(count = 1) {
    this.state = "ejecting";
    this.ejectAnim = 1.0;
    this.shakeTimer = 0.35 + Math.min(0.4, count * 0.08);
  }

  triggerPulse() {
    this.pulseAmount = Math.min(1.0, this.pulseAmount + 0.45);
  }

  _updateLed(elapsed) {
    if (!this.ledMat || !this.ledLight) return;
    let colorHex = 0xe2e8f0;
    let intensity = 0.45 + 0.15 * Math.sin(elapsed * 1.6);

    if (this.state === "scanning") {
      colorHex = 0x0d9488;
      intensity = 1.2 + 0.8 * Math.sin(elapsed * 6.5) + this.pulseAmount * 0.9;
    } else if (this.state === "ejecting") {
      colorHex = 0x14b8a6;
      intensity = 1.8 + 0.5 * Math.sin(elapsed * 4.0);
    } else if (this.state === "refused") {
      colorHex = 0xd97706;
      intensity = 1.1 + 0.95 * Math.abs(Math.sin(elapsed * 9.0));
    }

    this.ledMat.color.setHex(colorHex);
    this.ledMat.emissive.setHex(colorHex);
    this.ledMat.emissiveIntensity = intensity;
    this.ledLight.color.setHex(colorHex);
    this.ledLight.intensity = intensity * 0.55;
  }

  _loop(now) {
    if (!this.renderer || !this.scene || !this.camera) return;
    const dt = Math.min(0.08, Math.max(0.001, (now - this._lastTs) / 1000.0));
    this._lastTs = now;
    const elapsed = (now - this._startTs) / 1000.0;

    this.pulseAmount *= Math.exp(-dt * 5.0);

    const targetRotY = -0.22 + this.pointer.x * 0.14;
    const targetRotX = 0.16 + this.pointer.y * 0.08;
    const k = 1.0 - Math.exp(-dt * 6.0);
    this.rotSmooth.x += (targetRotX - this.rotSmooth.x) * k;
    this.rotSmooth.y += (targetRotY - this.rotSmooth.y) * k;

    let jitterX = 0;
    let jitterY = 0;
    if (this.shakeTimer > 0) {
      this.shakeTimer = Math.max(0, this.shakeTimer - dt);
      const amp = this.shakeTimer * 0.045;
      jitterX = Math.sin(elapsed * 58.0) * amp;
      jitterY = Math.cos(elapsed * 46.0) * amp * 0.6;
    }

    if (this.cartridgeGroup) {
      this.cartridgeGroup.rotation.x = this.rotSmooth.x;
      this.cartridgeGroup.rotation.y = this.rotSmooth.y;
      this.cartridgeGroup.position.set(
        jitterX,
        0.16 + Math.sin(elapsed * 1.4) * 0.025 + jitterY,
        0
      );
    }

    const spinSpeed =
      this.ejectAnim > 0.05
        ? 11.0
        : this.state === "scanning"
          ? 4.2
          : 0.25;
    if (this.rollerFront) this.rollerFront.rotation.x += dt * spinSpeed;
    if (this.rollerBack) this.rollerBack.rotation.x -= dt * spinSpeed;
    if (this.knobGroup) this.knobGroup.rotation.x += dt * spinSpeed * 0.65;

    if (this.filmTongue) {
      if (this.ejectAnim > 0) {
        this.ejectAnim = Math.max(0, this.ejectAnim - dt * 1.15);
        this.filmTongue.visible = true;
        const progress = 1.0 - this.ejectAnim;
        const ext = Math.sin(progress * Math.PI);
        this.filmTongue.position.y = -0.44 - ext * 0.85;
        this.filmTongue.rotation.x = ext * 0.16;
        this.filmTongue.material.color.setHex(this.state === "refused" ? 0x78350f : 0x0f2530);
      } else {
        this.filmTongue.visible = false;
      }
    }

    this._updateLed(elapsed);
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame((ts) => this._loop(ts));
  }
}
