/** Overview camera: fits the track's bounding box with 6% padding (SPEC 11.7). World y is up. */
export interface Camera {
  scale: number // device pixels per meter
  ox: number // screen x of world x = 0
  oy: number // screen y of world y = 0
}

/** Screen region (device px) the track should fit into; defaults to the whole canvas. */
export interface FitRegion {
  left: number
  top: number
  right: number
  bottom: number
}

export function fitCamera(
  centerline: Float32Array,
  widthM: number,
  canvasW: number,
  canvasH: number,
  padding = 0.06,
  region: FitRegion = { left: 0, top: 0, right: canvasW, bottom: canvasH },
): Camera {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity // prettier-ignore
  for (let i = 0; i < centerline.length; i += 2) {
    minX = Math.min(minX, centerline[i]!)
    maxX = Math.max(maxX, centerline[i]!)
    minY = Math.min(minY, centerline[i + 1]!)
    maxY = Math.max(maxY, centerline[i + 1]!)
  }
  const half = widthM / 2 + 2
  minX -= half
  minY -= half
  maxX += half
  maxY += half
  const regionW = region.right - region.left
  const regionH = region.bottom - region.top
  const availW = regionW * (1 - 2 * padding)
  const availH = regionH * (1 - 2 * padding)
  const scale = Math.min(availW / (maxX - minX), availH / (maxY - minY))
  return {
    scale,
    ox: region.left + regionW / 2 - ((minX + maxX) / 2) * scale,
    oy: region.top + regionH / 2 + ((minY + maxY) / 2) * scale,
  }
}

export const sx = (cam: Camera, x: number) => cam.ox + x * cam.scale
export const sy = (cam: Camera, y: number) => cam.oy - y * cam.scale
