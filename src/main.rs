use anyhow::Result;
use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};
//use rand::prelude::*;

fn main() -> Result<()> {
    launch::<_, GolCubeVisualizer>(Settings::default())
}

struct GolCubeVisualizer {
    verts: VertexBuffer,
    indices: IndexBuffer,
    points_shader: Shader,
    camera: MultiPlatformCamera,

    gol_cube: GolHypercube,
    //frame: usize,
}

impl App for GolCubeVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let mut gol_cube = GolHypercube::new(4, 25);

        let width = gol_cube.width();
        for face in gol_cube.front_data_mut() {
            for i in 0..width {
                face[(i, i)] = true;
                face[(width - i - 1, i)] = true;
            }
        }

        let vertices = inner_float_vertices(gol_cube.faces(), gol_cube.width(), 1.);
        let d3_inner_verts: Vec<Vertex> = vertices
            .into_iter()
            .map(|v| project_4_to_3(v, 0.3))
            .map(|pos| Vertex {
                pos,
                color: [1.; 3],
            })
            .collect();
        let indices = golcube_tri_indices(&gol_cube);

        Ok(Self {
            gol_cube,
            points_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Triangles,
            )?,
            verts: ctx.vertices(&d3_inner_verts, false)?,
            indices: ctx.indices(&indices, true)?,
            camera: MultiPlatformCamera::new(platform),
            //frame: 0,
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        let indices = golcube_tri_indices(&self.gol_cube);
        ctx.update_indices(self.indices, &indices)?;

        Ok(vec![DrawCmd::new(self.verts)
            .limit(indices.len() as _)
            .indices(self.indices)
            .shader(self.points_shader)])
    }

    fn event(
        &mut self,
        ctx: &mut Context,
        platform: &mut Platform,
        mut event: Event,
    ) -> Result<()> {
        if self.camera.handle_event(&mut event) {
            ctx.set_camera_prefix(self.camera.get_prefix())
        }
        match (event, platform) {
            (
                Event::Winit(idek::winit::event::Event::WindowEvent {
                    event: idek::winit::event::WindowEvent::CloseRequested,
                    ..
                }),
                Platform::Winit { control_flow, .. },
            ) => **control_flow = idek::winit::event_loop::ControlFlow::Exit,
            _ => (),
        }
        Ok(())
    }
}

const MAX_DIMS: usize = 4;
type DimensionBits = u8;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Face {
    /// Dimensions spanned by u, v
    pub u_dim: usize,
    pub v_dim: usize,
    /// The bits corresponding to this particular face
    pub bits: DimensionBits,
}

pub fn project_4_to_3([x, y, z, w, ..]: [f32; MAX_DIMS], scale: f32) -> [f32; 3] {
    [x, y, z].map(|v: f32| v * (1. - w * scale))
}

/// Float vertices for mesh rendering
pub fn inner_float_vertices(faces: &[Face], width: usize, scale: f32) -> Vec<[f32; MAX_DIMS]> {
    let idx_to_pos = |i: usize| scale * ((i as f32 / width as f32) * 2. - 1.);
    let mut output = vec![];
    for face in faces {
        let mut out = [0.0; MAX_DIMS];

        out.iter_mut()
            .zip(iter_bits_low_to_high(face.bits))
            .for_each(|(o, bit)| *o = if bit { -scale } else { scale });

        for v in 0..=width {
            out[face.v_dim] = idx_to_pos(v);
            for u in 0..=width {
                out[face.u_dim] = idx_to_pos(u);
                output.push(out);
            }
        }
    }
    output
}

/// Output with the given const size, but use the given value of m
pub fn n_choose_m(n: usize, m: usize) -> Vec<[usize; MAX_DIMS]> {
    assert!(m <= MAX_DIMS);

    let m_minus_one = match m.checked_sub(1) {
        Some(mmo) => mmo,
        None => return vec![[0; MAX_DIMS]],
    };

    let mut out = vec![];

    for i in m_minus_one..n {
        for mut sub in n_choose_m(i, m_minus_one) {
            sub[m_minus_one] = i;
            out.push(sub);
        }
    }
    out
}

fn check_bit(bits: DimensionBits, dim: usize) -> bool {
    (bits >> dim) & 1 == 1
}

/// Faces of the cube. Outputs the indices of the dimensions the face resides in, and the bit mask
/// representing which side
pub fn faces(n_dims: usize) -> Vec<Face> {
    let mut faces = vec![];
    for uv_dims in n_choose_m(n_dims, 2) {
        for perm in vertices(n_dims - 2) {
            let mut perm = iter_bits_low_to_high(perm);
            let mut bits: DimensionBits = 0;

            for idx in (0..n_dims).rev() {
                bits <<= 1;
                bits |= if idx == uv_dims[0] || idx == uv_dims[1] {
                    0
                } else {
                    perm.next().unwrap() as DimensionBits
                };
            }

            faces.push(Face {
                u_dim: uv_dims[0],
                v_dim: uv_dims[1],
                bits,
            });
        }
    }
    faces
}

/// Iterate over the bits of this u8, from low to high
fn iter_bits_low_to_high(mut v: u8) -> impl Iterator<Item = bool> {
    std::iter::from_fn(move || {
        let out = v & 0b00000001 != 0;
        v >>= 1;
        Some(out)
    })
    .take(u8::BITS as _)
}

/// An iterator over the vertices for a cube of the given number of dimensions
fn vertices(n_dims: usize) -> impl Iterator<Item = DimensionBits> {
    0..=DimensionBits::MAX >> (DimensionBits::BITS.checked_sub(n_dims as u32).unwrap())
}

fn golcube_tri_indices(cube: &GolHypercube) -> Vec<u32> {
    let mut indices = vec![];
    let idx_stride = cube.width as u32 + 1;

    let face_data_stride = cube.width * cube.width;
    let face_idx_stride = idx_stride * idx_stride;

    for (face_idx, face) in cube.front_data().iter().enumerate() {
        let mut backface = |[a, b, c]: [u32; 3]| {
            indices.extend_from_slice(&[a, b, c]);
            indices.extend_from_slice(&[c, b, a]);
        };

        let face_base = face_idx as u32 * face_idx_stride;
        for (y, row) in face.data().chunks_exact(cube.width).enumerate() {
            let row_base = face_base + y as u32 * idx_stride;
            for (x, &elem) in row.iter().enumerate() {
                let elem_idx = row_base + x as u32;
                if elem {
                    backface([elem_idx + idx_stride, elem_idx + 1, elem_idx]);

                    backface([
                        elem_idx + idx_stride,
                        elem_idx + idx_stride + 1,
                        elem_idx + 1,
                    ]);
                }
            }
        }
    }

    indices
}

pub struct GolHypercube {
    front: Vec<Square2DArray<bool>>,
    back: Vec<Square2DArray<bool>>,
    faces: Vec<Face>,
    width: usize,
}

impl GolHypercube {
    pub fn new(n_dims: usize, width: usize) -> Self {
        assert!(n_dims >= 3);

        let faces = faces(n_dims);

        let front: Vec<Square2DArray<bool>> = (0..faces.len())
            .map(|_| Square2DArray::new(width))
            .collect();
        let back = front.clone();

        Self {
            front,
            back,
            faces,
            width,
        }
    }

    pub fn faces(&self) -> &[Face] {
        &self.faces
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn front_data(&self) -> &[Square2DArray<bool>] {
        &self.front
    }

    pub fn front_data_mut(&mut self) -> &mut [Square2DArray<bool>] {
        &mut self.front
    }
}

#[derive(Clone)]
pub struct Square2DArray<T> {
    width: usize,
    data: Vec<T>,
}

impl<T> Square2DArray<T> {
    pub fn from_array(width: usize, data: Vec<T>) -> Self {
        assert!(data.len() % width == 0);
        assert!(data.len() / width == width);
        Self { width, data }
    }

    pub fn new(width: usize) -> Self
    where
        T: Default + Copy,
    {
        Self {
            width,
            data: vec![T::default(); width * width],
        }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    fn calc_index(&self, (x, y): (usize, usize)) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.width);
        x + y * self.width
    }
}

impl<T> std::ops::Index<(usize, usize)> for Square2DArray<T> {
    type Output = T;
    fn index(&self, pos: (usize, usize)) -> &T {
        &self.data[self.calc_index(pos)]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for Square2DArray<T> {
    fn index_mut(&mut self, pos: (usize, usize)) -> &mut T {
        let idx = self.calc_index(pos);
        &mut self.data[idx]
    }
}
