using CUDA
using BenchmarkTools
using Images
using ImageView
using LinearAlgebra
using StaticArrays

width = 576
height = 576
num_passes = 2000

function configurator(width,height,kernel)
    config = launch_configuration(kernel.fun)
    total_threads = Base.min(width*height,config.threads)
    threads_per_block_x = convert(Int32,floor(sqrt(total_threads)*(width/height)))
    threads_per_block_y = convert(Int32,floor(sqrt(total_threads)*(height/width)))
    blocks_x = cld(width, threads_per_block_x)
    blocks_y = cld(height, threads_per_block_y)
    return (threads_per_block_x, threads_per_block_y), (blocks_x, blocks_y)
end

struct Camera
    pos::SVector{3, Float32}
    dir::SVector{3, Float32}
    up::SVector{3, Float32}
    right::SVector{3, Float32}
    Camera(p,d,u) = new(p,normalize(d),normalize(u),normalize(cross(d,u)))
end

function lerp(vec1,vec2,α)
    return (α.*vec2) .+ (1 - α).*vec1
end

function smooth_step(edge0,edge1,x)
    x = clamp((x-edge0)/(edge1-edge0),0.0f0,1.0f0)
    return x*x*(3.0f0 - 2.0f0 * x)
end

function rand_norm_dist()
    θ = 2*π*rand(Float32)
    ρ = sqrt(-2*log(rand(Float32)))
    return ρ * cos(θ)
end

function unit_sphere_vec()
    v1 = rand_norm_dist()
    v2 = rand_norm_dist()
    v3 = rand_norm_dist()
    vec = SVector{3,Float32}(v1,v2,v3)
    return normalize(vec)
end

function length_squared(vec)::Float32
    return dot(vec,vec)
end

function max_float32()::Float32
    return Float32(2.0e31 - 1)
end

function rotation_matrix(o)
    Rx = [
          1 0 0;
          0 cos(o[1]) -sin(o[1]);
          0 sin(o[1]) cos(o[1])
        ]
    Ry = [
          cos(o[2]) 0 sin(o[2]);
          0 1 0;
          -sin(o[2]) 0 cos(o[2])
        ]    
    Rz = [
          cos(o[3]) -sin(o[3]) 0;
          sin(o[3]) cos(o[3]) 0;
          0 0 1
        ]
    return Rz*Ry*Rx
end

function rotate(vec,rot_mat)::SVector{3,Float32}
    return rot_mat*vec
end

function translate(vec,pos)::SVector{3,Float32}
    return vec + pos
end

function scale(vec,mag)::SVector{3,Float32}
    return mag*vec
end

function transform(vec,rot_mat,pos,mag)::SVector{3,Float32}
    return translate(scale(rotate(vec,rot_mat),mag),pos)
end

struct material 
    colour::NTuple{4, Float32}
    emission_colour::NTuple{4, Float32}
    roughness::Float32
    brightness::Float32
    refract_index::Float32
    transparency::Float32
    flag::UInt8

    function material(c::NTuple{4, Float32},e_c::NTuple{4, Float32},r::Float32,b::Float32,r_i::Float32,t::Float32,f::UInt8) #all params
       return new(c,e_c,r,b,r_i,t,f)
    end
    
    function material(colour::NTuple{4, Float32},roughness::Float32) #rough/metallic objects
        return new(colour,colour,roughness,0.0f0,0.0f0,0.0f0,0x00)
    end

    function material(colour::NTuple{4, Float32},emission_colour::NTuple{4, Float32},brightness::Float32) #light sources
        return new(colour,emission_colour,0.0f0,brightness,0.0f0,0.0f0,0x00)
    end

    function material()
        return new(NTuple{4, Float32}((0.0, 0.0, 0.0, 0.0)), NTuple{4, Float32}((0.0, 0.0, 0.0, 0.0)), 1.0f0, 0.0f0, 0.0f0, 0.0f0, 0x00) #no params
    end
end

struct sphere
    pos::SVector{3, Float32}
    radius::Float32
    mat::material
end

struct triangle
    posA::SVector{3, Float32}
    posB::SVector{3, Float32}
    posC::SVector{3, Float32}
    normal::SVector{3, Float32}

    function triangle(posA,posB,posC)
        edgeAB = posB - posA
        edgeAC = posC - posA
        normal = normalize(cross(edgeAB, edgeAC)) 
        return new(posA,posB,posC,normal)
    end

    function triangle(a,b,c,n)
        return new(a,b,c,n)
    end

    function triangle(posA,posB,posC,norm,pos,rot_mat,scale)
        A = transform(posA,rot_mat,pos,scale)
        B = transform(posB,rot_mat,pos,scale)
        C = transform(posC,rot_mat,pos,scale)
        normal = rotate(norm,rot_mat)
        return new(A,B,C,normal)
    end
end

struct mesh_info
    first_index::UInt32    #indexing begins at 1
    num_triangles::UInt32
    min_bounds::SVector{3, Float32}
    max_bounds::SVector{3, Float32}
    mat::material
end

function create_mesh(mat_type,tris,current_count,pos,orientation,scale)
    rot_mat = rotation_matrix((2*π/360)*orientation)
    
    minbound = @SVector [max_float32() for i=1:3]
    maxbound = @SVector [-max_float32() for i=1:3]
    
    new_tri_list = copy(tris)
    for (i,tri) in enumerate(tris)
        new_tri = triangle(tri.posA,tri.posB,tri.posC,tri.normal,pos,rot_mat,scale)
        new_tri_list[i] = new_tri
        
        minbound = min.(minbound,new_tri.posA)
        maxbound = max.(maxbound,new_tri.posA)
        minbound = min.(minbound,new_tri.posB)
        maxbound = max.(maxbound,new_tri.posB)
        minbound = min.(minbound,new_tri.posC)
        maxbound = max.(maxbound,new_tri.posC)
    end

    object_mesh = mesh_info(current_count,length(tris),minbound,maxbound,mat_type)
    return new_tri_list,object_mesh,current_count+length(tris)
end

struct hit_info
    did_hit::Bool
    is_front_face::Bool
    dist::Float32
    hit_pos::SVector{3, Float32}
    normal::SVector{3, Float32}
    mat::material

    function hit_info()
        return new(false,true,max_float32(),SVector{3,Float32}(0,0,0),SVector{3,Float32}(0,0,0),material())
    end

    function hit_info(h,f,d,p,n,m)
        return new(h,f,d,p,n,m)
    end
end

function get_normal(pos,obj::sphere)::SVector{3, Float32}
    return (pos - obj.pos)./obj.radius
end

function get_pos(origin, direction, dist)::SVector{3, Float32}
    return origin + dist*direction
end

function hits_box(ray_pos,ray_dir, boxMin, boxMax)::Bool	
    invDir = 1 ./ ray_dir
    tMin = (boxMin - ray_pos) .* invDir
    tMax = (boxMax - ray_pos) .* invDir
    t1 = min.(tMin, tMax)
    t2 = max.(tMin, tMax)
    tNear = max.(max.(t1[1], t1[2]), t1[3])
    tFar = min.(min.(t2[1], t2[2]), t2[3])
    return tNear <= tFar
end		

function intersect(ray_pos,ray_dir,obj::sphere)::hit_info
    ray_to_centre = ray_pos - obj.pos
    a = length_squared(ray_dir)
    half_b = dot(ray_to_centre,ray_dir)
    c = dot(ray_to_centre,ray_to_centre) - obj.radius * obj.radius
    discriminant = half_b * half_b - a * c;
    if discriminant > 0 
        sqrtd = sqrt(discriminant)
        dist1 = (-half_b - sqrtd) / a
        if dist1 >= 0 
            pos = get_pos(ray_pos,ray_dir,dist1)
            normal = get_normal(pos,obj)
            is_front = (dot(ray_dir,normal) < 0)
            return hit_info(true,is_front,dist1,pos,normal,obj.mat)
        else
            dist2 = (-half_b + sqrtd) / a
            if dist2 >= 0
                pos = get_pos(ray_pos,ray_dir,dist2)
                normal = get_normal(pos,obj)
                is_front = (dot(ray_dir,normal) < 0)
                return hit_info(true,is_front,dist2,pos,normal,obj.mat)
            end
        end
    end
    return hit_info()
end

function intersect(ray_pos,ray_dir,tri::triangle,mesh_material::material)::hit_info
    edgeAB = tri.posB - tri.posA
    edgeAC = tri.posC - tri.posA
    norm = cross(edgeAB, edgeAC)
    ao = ray_pos - tri.posA
    dao = cross(ao, ray_dir)

    determinant = Float32(-dot(ray_dir, norm))
    inv_det = Float32(1 / determinant)
    
    #Calculate dist to triangle & barycentric coordinates of intersection point
    dist = Float32(dot(ao, norm) * inv_det)
    u = Float32(dot(edgeAC, dao) * inv_det)
    v = Float32(-dot(edgeAB, dao) * inv_det)
    w = Float32(1 - u - v)
    
    if dist >= 0.0f0 && u >= 0.0f0 && v >= 0.0f0 && w >= 0.0f0
        is_front_face = (dot(tri.normal,ray_dir)<0)
        return hit_info(true,is_front_face,dist,get_pos(ray_pos,ray_dir,dist),tri.normal,mesh_material)
    else
        return hit_info()
    end
end

function ray_trace(scene,meshes,triangles,ray_origin,ray_direction)::hit_info
    current_hit = hit_info()
    for obj in scene
    	hitinfo = intersect(ray_origin,ray_direction,obj)
    	if hitinfo.did_hit == true && hitinfo.dist < current_hit.dist
    		current_hit = hitinfo
        end
    end
    
    for mesh in meshes
        if hits_box(ray_origin,ray_direction,mesh.min_bounds,mesh.max_bounds)
            for i in mesh.first_index : (mesh.first_index -1 + mesh.num_triangles) 
                @inbounds hitinfo = intersect(ray_origin,ray_direction,triangles[i],mesh.mat)
            	if hitinfo.did_hit == true && hitinfo.dist < current_hit.dist
                    current_hit = hitinfo
                end
            end
        end
    end
    
    return current_hit
end

function skycolour(ray_dir)::NTuple{4, Float32}
    
    sky_blue = NTuple{4, Float32}((0.7, 0.86, 1.0, 0.0))
    white = NTuple{4, Float32}((1.0, 1.0, 1.0, 0.0))
	a = 2*(ray_dir[2] + 1)
	return  lerp(white,sky_blue,a)
    
    #return NTuple{4, Float32}((0.0, 0.0, 0.0, 0.0))
end

function reflectance(cosθ,RI)::Float32 #chance for transparent object to reflect/scatter ray rather than refract 
    r0 = (1 - RI) / (1 + RI)
    r0_sq = r0*r0
    return r0_sq + (1-r0_sq)*((1 - cosθ)^5)
end

function reflect_ray(ray_in,normal)::SVector{3, Float32}
    return ray_in - normal * (2 * dot(ray_in,normal))
end

function diffuse_ray(normal)::SVector{3, Float32}
    return normalize(normal + unit_sphere_vec())
end

function refract_ray(r_in,normal,RI::Float32,is_front::Float32,cosθ::Float32)::SVector{3, Float32}
    normal = is_front*normal
    return normalize(RI*r_in + (-sqrt(abs(1-(RI^2)*(1-cosθ^2))) - RI*cosθ)*normal)
end
    
function pixel_colour(scene,meshes,triangles,ray_origin,ray_direction)::NTuple{4, Float32}
    safety_factor = 0.001f0
    num_bounces = 10
    absorbtion = NTuple{4, Float32}((1.0f0,1.0f0,1.0f0,0.0f0))
    origin = ray_origin
    direction = ray_direction
    colour_buffer = NTuple{4, Float32}((0.0f0, 0.0f0, 0.0f0, 0.0f0))
    for i in 1:num_bounces
        hitinfo = ray_trace(scene,meshes,triangles,origin,direction)
        if hitinfo.did_hit == true
            front_face = Float32((-1)^(1+hitinfo.is_front_face)) # 1 if is_front_face=true, -1 if false
            cosθ = dot(direction,-front_face*hitinfo.normal) #used to test refraction condition
            
            #refractive scattering            
            if (hitinfo.mat.transparency > rand(Float32)) && (reflectance(cosθ,hitinfo.mat.refract_index) < rand(Float32))
                sinθ = sqrt(1-cosθ^2)
                RI = hitinfo.mat.refract_index^(front_face) #invert relative RI if exiting material
                if sinθ > RI #TIR condition
                    specular_ray = reflect_ray(direction,hitinfo.normal)
                    direction = specular_ray
                    origin = hitinfo.hit_pos + front_face*safety_factor*hitinfo.normal
                else
                    refracted_ray = refract_ray(direction,hitinfo.normal,RI,front_face,cosθ)
                    direction = refracted_ray
                    origin = hitinfo.hit_pos + -front_face*safety_factor*hitinfo.normal  #safety factor is inverted when passing through surface
                end
            else
                matte_ray = diffuse_ray(hitinfo.normal)     #diffuse scattering
                specular_ray = reflect_ray(direction,hitinfo.normal)          #reflective scattering

                direction = lerp(specular_ray,matte_ray,hitinfo.mat.roughness) #linearly interpolate based on roughness
                origin = hitinfo.hit_pos + front_face*safety_factor*hitinfo.normal  
            end

            colour_buffer = colour_buffer .+ hitinfo.mat.brightness .* absorbtion .* hitinfo.mat.emission_colour 
            absorbtion = absorbtion.*hitinfo.mat.colour
        else
            colour_buffer = colour_buffer .+ absorbtion .* skycolour(ray_direction)
            break
        end
    end

    return colour_buffer .+ NTuple{4, Float32}((0.0f0,0.0f0,0.0f0,1.0f0))   #Set α==1
end

function imgfill!(img, width, height, num_passes, cam, scene, meshes, triangles)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= width && j <= height
        imgbuffer = NTuple{4, Float32}((0,0,0,0))
        for k in 1:num_passes
            u = Float32(-(i + (rand(Float32) - 0.5) - width / 2) / max(width, height))
            v = Float32((j + (rand(Float32) - 0.5) - height / 2) / max(width, height))      
            ray_direction = normalize(cam.dir + v*cam.right + u*cam.up)      
            imgbuffer = imgbuffer .+ pixel_colour(scene,meshes,triangles,cam.pos,ray_direction) 
        end
        imgbuffer = imgbuffer.*(255/num_passes) #Convert to [0,255) and average over number of passes
        imgbuffer = Base.clamp.(floor.(imgbuffer),0.0f0,255.0f0) #Clamp values for conversion to UInt8
        @inbounds img[i, j] = imgbuffer
    end
    return
end
          #[x, y, z] 
p = Float32[0, 0, -4]
d = Float32[0, 0, 1]
u = Float32[0, 1, 0]

cam = Camera(p,d,u)

matte_yellow = material(NTuple{4, Float32}((0.7, 0.8, 0.3, 0)),  1.0f0)
matte_magenta = material(NTuple{4, Float32}((0.9, 0.5, 0.65, 0)),1.0f0)
matte_cyan = material(NTuple{4, Float32}((0.2, 0.9, 0.75, 0)),   1.0f0)
matte_black = material(NTuple{4, Float32}((0.05, 0.05, 0.05, 0)),1.0f0)
matte_white = material(NTuple{4, Float32}((1.0, 1.0, 1.0, 0)),1.0f0)
mirror= material(NTuple{4, Float32}((0.95, 0.95, 0.95, 0)),       0.0f0)
grass = material(NTuple{4, Float32}((0.95, 0.95, 0.65, 0)), NTuple{4, Float32}((0.95, 0.95, 0.65, 0)), 1.0f0, 0.0f0, 1.0f0, 0.0f0, 0x0c)
glass = material(NTuple{4, Float32}((0.95, 0.95, 0.95, 0)), NTuple{4, Float32}((0.95, 0.95, 0.95, 0)), 1.0f0, 0.0f0, 1.5f0, 1.0f0, 0x0c)
white_light = material(NTuple{4, Float32}((1.0, 1.0, 1.0, 0)), NTuple{4, Float32}((1.0, 1.0, 1.0, 0)), 10.0f0)
check_blue_purple = material(NTuple{4, Float32}((0.2, 0.2, 1.0, 0)), NTuple{4, Float32}((0.95, 0.5, 0.9, 0)), 0.8f0, 0.0f0, 1.0f0, 0.0f0, 0x01)
rainbow = material(NTuple{4, Float32}((0.0, 0.0, 0.0, 0)), NTuple{4, Float32}((0.1, 0.1, 0.1, 0)), 1.0f0, 2.0f0, 1.0f0, 0.0f0, 0x02)
swirl = material(NTuple{4, Float32}((0.7, 0.8, 0.3, 0)), NTuple{4, Float32}((0.2, 0.9, 0.75, 0)), 1.0f0, 0.0f0, 1.0f0, 0.0f0, 0x03)

        #       position,            radius,  material
scene = [sphere(Float32[0,-5001,4],  5000.0f0,matte_cyan)
        ,sphere(Float32[110,110,110],40.0f0,  white_light)
        #,sphere(Float32[0,0,-2],0.6f0,  glass)
        ]
staticscene = CuArray{sphere}(scene)

triangle1 = triangle(Float32[0.0, -0.75, 0.25], Float32[0.0, 0.25, 1.25], Float32[1.0, 0.25, -0.75], Float32[0.81649655, -0.40824828, 0.40824828]) 
triangle2 = triangle(Float32[0.0, -0.75, 0.25], Float32[0.0, 0.25, 1.25], Float32[-1.0, 0.25, -0.75], Float32[-0.81649655, -0.40824828, 0.40824828]) 
triangle3 = triangle(Float32[0.0, -0.75, 0.25], Float32[1.0, 0.25, -0.75], Float32[-1.0, 0.25, -0.75], Float32[0.0, -0.70710677, -0.70710677]) 
triangle4 = triangle(Float32[0.0, 0.25, 1.25], Float32[1.0, 0.25, -0.75], Float32[-1.0, 0.25, -0.75], Float32[0.0, 1.0, 0.0])
primitive_tetrahedron = [triangle1,triangle2,triangle3,triangle4]

count = 1
tetrahedron1,mesh1,count = create_mesh(matte_magenta,primitive_tetrahedron,count,Float32[0.7,0,0],Float32[0,90,45],0.75f0)
tetrahedron2,mesh2,count = create_mesh(matte_yellow,primitive_tetrahedron,count,Float32[-0.7,0,0],Float32[35,10,0],0.5f0)
tetrahedron3,mesh3,count = create_mesh(glass,primitive_tetrahedron,count,Float32[0,0,0],Float32[45,10,30],2.0f0)

triangle_list = vcat(tetrahedron1
                    ,tetrahedron2
                    ,tetrahedron3
                    )
mesh_list = [mesh1
            ,mesh2
            ,mesh3
            ]
triangles = CuArray{triangle}(triangle_list)
meshes = CuArray{mesh_info}(mesh_list)

CUimg = CUDA.fill(NTuple{4, UInt8}((0, 0, 0, 0)), width, height)
thr,blk = configurator(width,height,@cuda launch=false imgfill!(CUimg,width,height,num_passes,cam,staticscene,meshes,triangles))
CUDA.pool_status()

CUDA.synchronize()
CUDA.@time begin
    @cuda threads=thr blocks=blk imgfill!(CUimg, width, height, num_passes, cam, staticscene,meshes,triangles)
    CUDA.synchronize()
end
result_img = reinterpret(RGBA{N0f8}, Array(CUimg))