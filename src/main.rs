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
        let mut gol_cube = GolHypercube::new(3, 25);

        let width = gol_cube.width();
        let faces = gol_cube.faces.clone();

        let (u, v) = (0, 0);
        for du in -1..=1 {
            for dv in -1..=1 {
                gol_cube.overindex_set(u + du, v + dv, 0, true);
            }
        }

        /*
        for face_idx in 0..faces.len() {
            for i in 0..width as i32 {
                gol_cube.overindex_set(i, 2, face_idx, true);
                if i % 2 == 0 {
                    gol_cube.overindex_set(2, i, face_idx, true);
                }
                /*face[(i, 2)] = true;
                if i % 2 == 0 {
                    face[(2, i)] = true;
                }
                */
                /*
                face[(i, i)] = true;
                face[(width - i - 1, i)] = true;
                */
            }
        }
        */

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
            .for_each(|(o, bit)| *o = if bit { scale } else { -scale });

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

    pub fn overindex<'a>(&'a self, u: i32, v: i32, face_sel: usize) -> impl Iterator<Item=bool> + 'a {
        let indices = overindex_face(u, v, face_sel, &self.faces, self.width);
        indices.into_iter().map(move |(face_idx, (u, v))| self.front[face_idx][(u as usize, v as usize)])
    }

    pub fn overindex_set<'a>(&'a mut self, u: i32, v: i32, face_sel: usize, set: bool) {
        let indices = overindex_face(u, v, face_sel, &self.faces, self.width);
        indices.into_iter().for_each(move |(face_idx, (u, v))| self.front[face_idx][(u as usize, v as usize)] = set)
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

fn extended_gol_rules(center: bool, neighbors: u32) -> bool {
    match (center, neighbors) {
        (true, n) if (n == 2 || n == 3) => true,
        (false, n) if (n == 3) => true,
        _ => false,
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

/// Index into a face `face_sel` from faces, possibly beyond width (in which case possibly several faces can be returned!
pub fn overindex_face(
    u: i32,
    v: i32,
    face_sel: usize,
    faces: &[Face],
    width: usize,
) -> Vec<(usize, (i32, i32))> {
    let in_bounds = |x: i32| x < 0 || x >= width as i32;

    match (in_bounds(u), in_bounds(v)) {
        // Indexing past a corner
        (true, true) => vec![],
        // In bounds
        (false, false) => vec![(face_sel, (u, v))],
        // Out of bounds
        (u_out_of_bounds, _) => {
            // Find all faces which:
            // * Index using the in bounds dim
            // * Don't index using the out of bounds dim
            // * Have the bit corresponding to the out of bounds dim set to
            // out_bounds_sign(out_of_bounds_dim)
            // * And index into that face so that the in bounds dim is the same, and the out of
            // bounds dim is towards the original face
            let overindexed_face = faces[face_sel];
            let (out_bounds_dim, in_bounds_dim, out_bounds_sign, in_bounds_val) =
                match u_out_of_bounds {
                    true => (overindexed_face.u_dim, overindexed_face.v_dim, u > 0, v),
                    false => (overindexed_face.v_dim, overindexed_face.u_dim, v > 0, u),
                };

            // For each other face
            faces
                .iter()
                .enumerate()
                .filter_map(|(idx, face)| {
                    // Check that the bit for the dims of the given face is set the same as the out of bounds dim's sign
                    if check_bit(face.bits, out_bounds_dim) != out_bounds_sign {
                        return None;
                    }
                    //dbg!(face.u_dim, face.v_dim, in_bounds_dim, out_bounds_dim, out_bounds_sign);

                    // Make sure the only dimensions/bits that change are the ones involved
                    let check_mask = (1 << out_bounds_dim) | (1 << face.u_dim) | (1 << face.v_dim);
                    if !check_mask & (face.bits ^ overindexed_face.bits) != 0 {
                        return None;
                    }

                    // Construct indexing information using the out of bounds sign, and the in
                    // bounds position
                    if face.u_dim == in_bounds_dim && face.v_dim != out_bounds_dim {
                        Some((
                            idx,
                            (
                                in_bounds_val,
                                if check_bit(overindexed_face.bits, face.v_dim) {
                                    width as _
                                } else {
                                    0
                                },
                            ),
                        ))
                    } else if face.v_dim == in_bounds_dim && face.u_dim != out_bounds_dim {
                        Some((
                            idx,
                            (
                                if check_bit(overindexed_face.bits, face.u_dim) {
                                    width as _
                                } else {
                                    0
                                },
                                in_bounds_val,
                            ),
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}
