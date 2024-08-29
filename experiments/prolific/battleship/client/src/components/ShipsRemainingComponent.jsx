import React, { useState, useEffect } from 'react';
import Board from '../stages/Board.js';
import { getColor } from '../../../utils/colorUtils.js';
import { useRound } from '@empirica/core/player/classic/react';

export function ShipsRemainingComponent({shipsStatus}) { //expects an array of the form [[1,[true,2]],[2,[false,3]],...]
  
  const round = useRound();

  function sortShipsLength(arr) {
    return arr.sort((a, b) => a[1][1] - b[1][1]);
  };
  
  function sortColors(arr) {
    return arr.sort((a, b) => a[0] - b[0]);
  };

  const [statuses, setStatuses] = useState(shipsStatus);

  return (
    <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
      <u style={{fontSize: "1.25vw", marginBottom:"0.2vw"}}>Ship Tracker</u>
      <div style={{display: "flex", flexDirection: "row", alignItems: "center", marginBottom:"0.7vw"}}>
      <i style={{fontSize: "0.8vw", marginRight:"0.2vw"}}>Ship Colors: </i> {sortColors(statuses).map((status) =>
      (
          <div
            style={{
              width: "1.5vw",
              height: "1.5vw",
              border: '1px solid white',
              opacity: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: getColor(status[0]),
              marginLeft:"0.1vw",
            }}
          >
          </div>
        ))
    }
      </div>

      <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'left',
              justifyContent: 'left'}}>
      {sortShipsLength(statuses).map((status) =>
      (
          <div
            style={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'left',
              color: 'white'
            }}
          >
            <svg width="4vw" height="4vw" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" ><path fill={status[1][0] ? getColor(status[0]) : 'gray'} d="M247 32v23h-23v18h23v22h-39v18h39v22h-46.027l-2.54 10.154-18.408-9.205-8.05 16.102 21.988 10.994-10.578 42.312-25.592 13.96 23.691 110.564c-49.074 3.341-98.15 8.946-145.6 16.453l2.813 17.777a1639.075 1639.075 0 0 1 121.096-14.529l-.078.31c112.547 28.156 190.551 43.088 306.816-8.958l-7.355-16.43a485.085 485.085 0 0 1-26.844 11.092c-32.405-4.352-66.372-7.09-101.246-8.381l23.121-107.899-25.592-13.959-10.578-42.312 21.988-10.994-8.05-16.102-18.409 9.205L311.027 135H265v-22h39V95h-39V73h23V55h-23V32h-18zm-31.973 121h81.946l10.16 40.639L256 165.748l-51.133 27.89L215.027 153zM256 186.252v140.346c-5.41.103-10.833.238-16.262.402h-40.46l-21.071-98.316L256 186.252zM224 208a16 16 0 0 0-16 16 16 16 0 0 0 16 16 16 16 0 0 0 16-16 16 16 0 0 0-16-16zm68.17 0a16 16 0 0 1 16 16 16 16 0 0 1-16 16 16 16 0 0 1-16-16 16 16 0 0 1 16-16zm-50.92 137h82.404c22.502.709 44.618 2.01 66.149 3.96-58.924 14.561-109.381 9.793-169.532-3.194A1523.33 1523.33 0 0 1 241.25 345zm-132.865 29.363c-7.943-.023-15.667.234-23.084.842l1.469 17.941c54.39-4.455 133.014 12.49 189.199 17.202 55.64 4.665 109.966-1.684 168.654-13.512l-3.557-17.645c-57.8 11.65-110.279 17.692-163.591 13.221-47.153-3.954-113.49-17.885-169.09-18.049zm20.22 35.285c-12.198-.079-25.387.615-38.517 1.873-26.26 2.518-51.6 7.157-67.865 14.26l7.203 16.496c12.302-5.372 37.244-10.427 62.38-12.838 25.138-2.41 51.157-2.311 65.846.625 32.956 6.589 91.409 16.938 138.62 15.444l-.569-17.99c-44.053 1.394-102.073-8.619-134.523-15.106-9.17-1.833-20.376-2.684-32.575-2.764z"/></svg>
            {Array.from({ length: status[1][1] }).map((_, idx) => (
            <div
              key={idx}
              style={{
                width: "1vw",
                height: "1vw",
                border: '1px solid white',
                opacity: '100%',
                display: 'flex',
                backgroundColor: status[1][0] ? getColor(status[0]) : 'gray',
                marginLeft: "0.1vw"
              }}
            />
          ))}

          </div>
        ))
    }
    </div>

    </div>
  );
}

export default ShipsRemainingComponent;
