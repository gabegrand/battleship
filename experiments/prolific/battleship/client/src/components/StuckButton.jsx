import React from "react";

const base =
  "inline-flex items-center px-4 py-2 border text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-empirica-500";
const prim =
  "border-gray-300 shadow-sm text-gray bg-darkgray hover:bg-darkgray";
const sec =
  "border-transparent shadow-sm text-white bg-empirica-600 hover:bg-empirica-700";

export function StuckButton({
  children,
  handleClick = null,
  className = "",
  primary = false,
  type = "button",
  autoFocus = false,
}) {
  return (
    <button
      type={type}
      onClick={handleClick}
      className={`${base} ${primary ? prim : sec} ${className}`}
      autoFocus={autoFocus}
      style={{borderRadius:"20px", fontSize:"1vw", height:"2vw", width:"10vw", opacity:"70%", backgroundColor:"darkred", textAlign:"center", justifyContent:"center", marginTop:"1vh"}}
    >
      {children}
    </button>
  );
}
