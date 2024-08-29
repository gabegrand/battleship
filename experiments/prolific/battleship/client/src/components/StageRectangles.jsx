import React from "react";

const baseStyle = "inline-flex items-center justify-center px-4 py-2 border text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-empirica-500";
const inactiveStyle = "border-gray-300 shadow-sm text-gray-700 bg-white";
const activeStyle = "border-transparent shadow-sm text-white bg-empirica-600";

export function StageRectangles({ signal }) {
  return (
    <div className="flex space-x-4" style={{marginTop:"10px"}}>
      <div
        className={`${baseStyle} ${signal === 0 ? activeStyle : inactiveStyle}`}
        style={{ width: "5vw", height: "2.5vw", borderStyle:"dashed", fontSize:"0.8vw"}}
      >
        Question
      </div>
      <div
        className={`${baseStyle} ${signal === 1 ? activeStyle : inactiveStyle}`}
        style={{ width: "5vw", height: "2.5vw", borderStyle:"dashed", fontSize:"0.8vw"}}
      >
        Answer 
      </div>
      <div
        className={`${baseStyle} ${signal === 2 ? activeStyle : inactiveStyle}`}
        style={{ width: "5vw", height: "2.5vw", borderStyle:"solid", fontSize:"0.8vw"}}
      >
        Fire! 
      </div>
    </div>
  );
}