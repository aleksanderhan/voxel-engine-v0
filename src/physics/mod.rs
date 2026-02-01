// src/physics/mod.rs
pub mod player;
pub mod collision;
pub mod step;
pub mod world_query;

pub use player::{PlayerBody, PlayerTuning};
pub use collision::WorldQuery;
pub use step::Physics;
pub use world_query::ChunkManagerQuery;

// Physics module.
//
// Starts with a character-style voxel collider + a minimal articulated-body
// (Featherstone articulated-body algorithm, ABA) scaffold.

pub mod spatial;
pub mod articulated;
pub mod voxel;
pub mod character;

pub use articulated::{ArticulatedBody, Joint, JointType, RigidBodyInertia};
pub use voxel::{StaticVoxelQuery, StaticVoxelWorldQuery};
pub use character::{CharacterController, CharacterState};

pub mod projectiles;
pub use projectiles::{DynamicVoxel, DynamicVoxelTuning};
