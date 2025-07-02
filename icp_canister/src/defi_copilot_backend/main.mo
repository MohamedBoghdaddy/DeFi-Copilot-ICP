import Nat "mo:base/Nat";
import Text "mo:base/Text";
import Debug "mo:base/Debug";
import HashMap "mo:base/HashMap";
import Principal "mo:base/Principal";

actor {

  // Define a UserProfile type
  type UserProfile = {
    name: Text;
    age: Nat;
    employmentStatus: Text;
    salary: Nat;
    riskTolerance: Text;
    goals: [Text];
  };

  // Storage: Map Principal -> Profile
  let profiles = HashMap.HashMap<Principal, UserProfile>(10, Principal.equal, Principal.hash);

  // Update or Create a profile
  public func saveProfile(profile: UserProfile): async Text {
    let caller = Principal.fromActor(this);
    let user = Principal.fromCaller();

    profiles.put(user, profile);
    Debug.print("✅ Profile saved for user: " # Principal.toText(user));
    return "Profile saved successfully!";
  };

  // Retrieve the profile of the caller
  public query func getMyProfile(): async ?UserProfile {
    let user = Principal.fromCaller();
    return profiles.get(user);
  };

  // Admin/debug: get profile of any principal (optional)
    return profiles.get(p);
  };

};
